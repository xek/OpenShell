// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Credential refresh trait and implementations for providers that issue
//! short-lived tokens (OAuth2, IAM, etc.).
//!
//! The gateway background refresh loop calls [`needs_refresh`] on each provider
//! record and, when true, calls [`refresh`] to obtain new credentials. Updated
//! credentials are written back to the provider store. The sandbox picks up the
//! new token on its next [`GetInferenceBundle`] poll — no restart required.

use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};
use tracing::{info, warn};

use crate::persistence::Store;

// ---------------------------------------------------------------------------
// Trait
// ---------------------------------------------------------------------------

/// Refresh short-lived credentials for a specific provider type.
#[async_trait::async_trait]
pub trait CredentialRefresher: Send + Sync {
    /// The provider type this refresher handles (e.g. `"vertex"`).
    fn provider_type(&self) -> &'static str;

    /// Return true if the credentials need to be refreshed now.
    ///
    /// Implementations should check a stored expiry timestamp and return true
    /// when fewer than [`REFRESH_BUFFER_SECS`] remain before expiry, or when
    /// no valid token is present.
    fn needs_refresh(&self, credentials: &HashMap<String, String>) -> bool;

    /// Exchange stored credentials for a fresh token.
    ///
    /// Returns the updated credential map. Only the keys that changed need to
    /// be returned — the caller merges them into the existing map.
    async fn refresh(
        &self,
        credentials: &HashMap<String, String>,
    ) -> Result<HashMap<String, String>, RefreshError>;
}

/// Seconds before expiry at which a refresh is triggered proactively.
const REFRESH_BUFFER_SECS: i64 = 300; // 5 minutes

#[derive(Debug, thiserror::Error)]
pub enum RefreshError {
    #[error("missing required credential: {0}")]
    MissingCredential(String),
    #[error("invalid service account JSON: {0}")]
    InvalidServiceAccount(String),
    #[error("JWT signing failed: {0}")]
    JwtSigningFailed(String),
    #[error("token endpoint request failed: {0}")]
    TokenRequestFailed(String),
    #[error("token endpoint returned unexpected response: {0}")]
    TokenResponseInvalid(String),
}

// ---------------------------------------------------------------------------
// Vertex AI refresher
// ---------------------------------------------------------------------------

/// Exchanges a Google service account JSON key for a short-lived OAuth2 Bearer
/// token using the JWT-based service account flow (RFC 7523).
pub struct VertexRefresher {
    client: reqwest::Client,
}

impl VertexRefresher {
    pub fn new() -> Self {
        Self {
            client: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(15))
                .build()
                .expect("failed to build HTTP client for VertexRefresher"),
        }
    }
}

impl Default for VertexRefresher {
    fn default() -> Self {
        Self::new()
    }
}

/// Parsed fields from a Google service account JSON key file.
#[derive(serde::Deserialize)]
struct ServiceAccount {
    client_email: String,
    private_key: String,
    #[serde(default = "default_token_uri")]
    token_uri: String,
}

/// Parsed fields from an ADC authorized_user credential file.
#[derive(serde::Deserialize)]
struct AuthorizedUser {
    client_id: String,
    client_secret: String,
    refresh_token: String,
    #[serde(default = "default_token_uri")]
    token_uri: String,
}

fn default_token_uri() -> String {
    "https://oauth2.googleapis.com/token".to_string()
}

#[derive(serde::Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum GoogleCredential {
    ServiceAccount(ServiceAccount),
    AuthorizedUser(AuthorizedUser),
}

#[derive(serde::Deserialize)]
struct TokenResponse {
    access_token: String,
    expires_in: i64,
}

#[async_trait::async_trait]
impl CredentialRefresher for VertexRefresher {
    fn provider_type(&self) -> &'static str {
        "vertex"
    }

    fn needs_refresh(&self, credentials: &HashMap<String, String>) -> bool {
        // If no access token yet, always refresh.
        if !credentials.contains_key("VERTEX_ACCESS_TOKEN") {
            return true;
        }
        // Check stored expiry timestamp (unix seconds).
        let Some(expires_at) = credentials.get("VERTEX_TOKEN_EXPIRES_AT") else {
            return true;
        };
        let Ok(expires_at) = expires_at.parse::<i64>() else {
            return true;
        };
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64;
        now >= expires_at - REFRESH_BUFFER_SECS
    }

    async fn refresh(
        &self,
        credentials: &HashMap<String, String>,
    ) -> Result<HashMap<String, String>, RefreshError> {
        let cred_json = credentials
            .get("GOOGLE_SERVICE_ACCOUNT_JSON")
            .ok_or_else(|| {
                RefreshError::MissingCredential("GOOGLE_SERVICE_ACCOUNT_JSON".to_string())
            })?;

        let cred: GoogleCredential = serde_json::from_str(cred_json)
            .map_err(|e| RefreshError::InvalidServiceAccount(e.to_string()))?;

        let token_resp = match cred {
            GoogleCredential::ServiceAccount(sa) => self.refresh_service_account(&sa).await?,
            GoogleCredential::AuthorizedUser(user) => self.refresh_authorized_user(&user).await?,
        };

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64;

        let mut updated = HashMap::new();
        // ANTHROPIC_API_KEY lets Claude Code authenticate when ANTHROPIC_BASE_URL
        // points at the cluster inference proxy (inference.local), which substitutes
        // this token on outbound Vertex requests.
        updated.insert(
            "ANTHROPIC_API_KEY".to_string(),
            token_resp.access_token.clone(),
        );
        updated.insert("VERTEX_ACCESS_TOKEN".to_string(), token_resp.access_token);
        updated.insert(
            "VERTEX_TOKEN_EXPIRES_AT".to_string(),
            (now + token_resp.expires_in).to_string(),
        );
        Ok(updated)
    }
}

impl VertexRefresher {
    async fn refresh_service_account(
        &self,
        sa: &ServiceAccount,
    ) -> Result<TokenResponse, RefreshError> {
        let jwt = build_jwt(sa)?;
        let params = [
            ("grant_type", "urn:ietf:params:oauth:grant-type:jwt-bearer"),
            ("assertion", jwt.as_str()),
        ];
        self.post_token(&sa.token_uri, &params).await
    }

    async fn refresh_authorized_user(
        &self,
        user: &AuthorizedUser,
    ) -> Result<TokenResponse, RefreshError> {
        let params = [
            ("grant_type", "refresh_token"),
            ("refresh_token", user.refresh_token.as_str()),
            ("client_id", user.client_id.as_str()),
            ("client_secret", user.client_secret.as_str()),
        ];
        self.post_token(&user.token_uri, &params).await
    }

    async fn post_token(
        &self,
        token_uri: &str,
        params: &[(&str, &str)],
    ) -> Result<TokenResponse, RefreshError> {
        let resp = self
            .client
            .post(token_uri)
            .form(params)
            .send()
            .await
            .map_err(|e| RefreshError::TokenRequestFailed(e.to_string()))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(RefreshError::TokenRequestFailed(format!(
                "HTTP {status}: {body}"
            )));
        }

        resp.json()
            .await
            .map_err(|e| RefreshError::TokenResponseInvalid(e.to_string()))
    }
}

/// Build a signed RS256 JWT for the Google OAuth2 service account flow.
fn build_jwt(sa: &ServiceAccount) -> Result<String, RefreshError> {
    use base64::{Engine as _, engine::general_purpose::URL_SAFE_NO_PAD};

    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64;

    // Header
    let header = serde_json::json!({"alg": "RS256", "typ": "JWT"});
    let header_b64 = URL_SAFE_NO_PAD.encode(header.to_string().as_bytes());

    // Claims
    let claims = serde_json::json!({
        "iss": sa.client_email,
        "scope": "https://www.googleapis.com/auth/cloud-platform",
        "aud": sa.token_uri,
        "iat": now,
        "exp": now + 3600,
    });
    let claims_b64 = URL_SAFE_NO_PAD.encode(claims.to_string().as_bytes());

    let signing_input = format!("{header_b64}.{claims_b64}");

    // Parse the PEM private key and sign with RS256 via ring.
    let der = pem_to_der(&sa.private_key)
        .map_err(|e| RefreshError::InvalidServiceAccount(format!("PEM decode failed: {e}")))?;

    let key_pair = ring::signature::RsaKeyPair::from_pkcs8(&der)
        .map_err(|e| RefreshError::JwtSigningFailed(format!("key parse failed: {e}")))?;

    let rng = ring::rand::SystemRandom::new();
    let mut signature = vec![0u8; key_pair.public().modulus_len()];
    key_pair
        .sign(
            &ring::signature::RSA_PKCS1_SHA256,
            &rng,
            signing_input.as_bytes(),
            &mut signature,
        )
        .map_err(|e| RefreshError::JwtSigningFailed(format!("signing failed: {e}")))?;

    let sig_b64 = URL_SAFE_NO_PAD.encode(&signature);
    Ok(format!("{signing_input}.{sig_b64}"))
}

/// Strip PEM armor and base64-decode to DER bytes.
///
/// Handles both `BEGIN PRIVATE KEY` (PKCS#8) and `BEGIN RSA PRIVATE KEY`
/// (traditional) formats. Google service account keys use PKCS#8.
fn pem_to_der(pem: &str) -> Result<Vec<u8>, String> {
    use base64::{Engine as _, engine::general_purpose::STANDARD};

    // Unescape literal `\n` sequences that appear in JSON-encoded PEM strings.
    let pem = pem.replace("\\n", "\n");

    let body: String = pem
        .lines()
        .filter(|l| !l.starts_with("-----"))
        .collect::<Vec<_>>()
        .join("");

    STANDARD
        .decode(body.trim())
        .map_err(|e| format!("base64 decode error: {e}"))
}

// ---------------------------------------------------------------------------
// Refresh loop
// ---------------------------------------------------------------------------

/// Spawn a background task that periodically refreshes short-lived provider
/// credentials for all registered refreshers.
///
/// The loop runs every `interval` seconds. For each provider record whose type
/// matches a refresher and whose [`CredentialRefresher::needs_refresh`] returns
/// true, the refresher is called and the updated credentials are written back
/// to the store.
pub fn spawn_credential_refresh_loop(store: std::sync::Arc<Store>, interval: std::time::Duration) {
    let refreshers: Vec<Box<dyn CredentialRefresher>> = vec![Box::new(VertexRefresher::new())];

    tokio::spawn(async move {
        // Initial delay to let server startup settle.
        tokio::time::sleep(interval).await;

        loop {
            run_refresh_sweep(&store, &refreshers).await;
            tokio::time::sleep(interval).await;
        }
    });
}

async fn run_refresh_sweep(store: &Store, refreshers: &[Box<dyn CredentialRefresher>]) {
    use openshell_core::proto::Provider;
    use prost::Message;

    let records = match store.list("provider", 1000, 0).await {
        Ok(r) => r,
        Err(e) => {
            warn!(error = %e, "credential refresh: failed to list providers");
            return;
        }
    };

    for record in records {
        let provider = match Provider::decode(record.payload.as_slice()) {
            Ok(p) => p,
            Err(_) => continue,
        };

        let Some(refresher) = refreshers
            .iter()
            .find(|r| r.provider_type() == provider.r#type.as_str())
        else {
            continue;
        };

        if !refresher.needs_refresh(&provider.credentials) {
            continue;
        }

        info!(
            provider = %provider.name,
            provider_type = %provider.r#type,
            "credential refresh: refreshing token"
        );

        match refresher.refresh(&provider.credentials).await {
            Ok(updated) => {
                let mut provider = provider;
                provider.credentials.extend(updated);
                if let Err(e) = store.put_message(&provider).await {
                    warn!(
                        error = %e,
                        provider = %provider.name,
                        "credential refresh: failed to persist updated credentials"
                    );
                } else {
                    info!(
                        provider = %provider.name,
                        "credential refresh: token updated"
                    );
                }
            }
            Err(e) => {
                warn!(
                    error = %e,
                    provider = %provider.name,
                    "credential refresh: token refresh failed"
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn needs_refresh_when_no_token() {
        let refresher = VertexRefresher::new();
        let creds = HashMap::new();
        assert!(refresher.needs_refresh(&creds));
    }

    #[test]
    fn needs_refresh_when_token_expiring_soon() {
        let refresher = VertexRefresher::new();
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64;
        let mut creds = HashMap::new();
        creds.insert("VERTEX_ACCESS_TOKEN".to_string(), "tok".to_string());
        // Expires in 2 minutes — within the 5-minute buffer.
        creds.insert(
            "VERTEX_TOKEN_EXPIRES_AT".to_string(),
            (now + 120).to_string(),
        );
        assert!(refresher.needs_refresh(&creds));
    }

    #[test]
    fn does_not_need_refresh_when_token_fresh() {
        let refresher = VertexRefresher::new();
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64;
        let mut creds = HashMap::new();
        creds.insert("VERTEX_ACCESS_TOKEN".to_string(), "tok".to_string());
        // Expires in 30 minutes — well outside the 5-minute buffer.
        creds.insert(
            "VERTEX_TOKEN_EXPIRES_AT".to_string(),
            (now + 1800).to_string(),
        );
        assert!(!refresher.needs_refresh(&creds));
    }

    #[test]
    fn pem_to_der_strips_armor_and_decodes() {
        // Minimal valid base64 body to confirm stripping works.
        let pem = "-----BEGIN PRIVATE KEY-----\naGVsbG8=\n-----END PRIVATE KEY-----\n";
        let der = pem_to_der(pem).expect("should decode");
        assert_eq!(der, b"hello");
    }

    #[test]
    fn pem_to_der_handles_escaped_newlines() {
        let pem = "-----BEGIN PRIVATE KEY-----\\naGVsbG8=\\n-----END PRIVATE KEY-----\\n";
        let der = pem_to_der(pem).expect("should decode escaped newlines");
        assert_eq!(der, b"hello");
    }

    /// Live integration test against real Vertex AI.
    ///
    /// Uses credentials from (in priority order):
    ///   GOOGLE_SERVICE_ACCOUNT_JSON  - inline JSON or path to credential file
    ///   GOOGLE_APPLICATION_CREDENTIALS - path to credential file
    ///   ~/.config/gcloud/application_default_credentials.json (ADC)
    ///
    /// Endpoint constructed from:
    ///   GOOGLE_CLOUD_PROJECT (or VERTEXAI_PROJECT / ANTHROPIC_VERTEX_PROJECT_ID)
    ///   VERTEXAI_LOCATION (or VERTEX_LOCATION), defaults to us-central1
    ///   VERTEX_MODEL, defaults to claude-3-5-haiku@20241022
    ///
    /// Run with:
    ///   cargo test -p openshell-server vertex_live -- --ignored --nocapture
    #[tokio::test]
    #[ignore]
    async fn vertex_live_refresh_and_infer() {
        // Resolve credentials from env or ADC well-known file.
        let cred_json = if let Ok(v) = std::env::var("GOOGLE_SERVICE_ACCOUNT_JSON") {
            if v.trim_start().starts_with('{') {
                v
            } else {
                std::fs::read_to_string(&v)
                    .unwrap_or_else(|_| panic!("could not read credential file: {v}"))
            }
        } else if let Ok(path) = std::env::var("GOOGLE_APPLICATION_CREDENTIALS") {
            std::fs::read_to_string(&path)
                .unwrap_or_else(|_| panic!("could not read GOOGLE_APPLICATION_CREDENTIALS: {path}"))
        } else {
            let home = std::env::var("HOME").expect("HOME must be set");
            let adc = format!("{home}/.config/gcloud/application_default_credentials.json");
            std::fs::read_to_string(&adc)
                .unwrap_or_else(|_| panic!("no credentials found; run `gcloud auth application-default login` or set GOOGLE_SERVICE_ACCOUNT_JSON"))
        };

        // Construct the rawPredict endpoint from project + location + model.
        let project = std::env::var("GOOGLE_CLOUD_PROJECT")
            .or_else(|_| std::env::var("VERTEXAI_PROJECT"))
            .or_else(|_| std::env::var("ANTHROPIC_VERTEX_PROJECT_ID"))
            .expect("set GOOGLE_CLOUD_PROJECT, VERTEXAI_PROJECT, or ANTHROPIC_VERTEX_PROJECT_ID");
        let location = std::env::var("VERTEXAI_LOCATION")
            .or_else(|_| std::env::var("VERTEX_LOCATION"))
            .unwrap_or_else(|_| "us-central1".to_string());
        let model = std::env::var("VERTEX_MODEL")
            .unwrap_or_else(|_| "claude-3-5-haiku@20241022".to_string());
        let base_url = format!(
            "https://{location}-aiplatform.googleapis.com/v1/projects/{project}/locations/{location}/publishers/anthropic/models/{model}:rawPredict"
        );
        println!("endpoint: {base_url}");

        // Step 1: refresh token.
        let refresher = VertexRefresher::new();
        let mut creds = HashMap::new();
        creds.insert("GOOGLE_SERVICE_ACCOUNT_JSON".to_string(), cred_json);

        assert!(
            refresher.needs_refresh(&creds),
            "should need refresh — no token yet"
        );

        let updated = refresher
            .refresh(&creds)
            .await
            .expect("token refresh should succeed");

        let token = updated
            .get("VERTEX_ACCESS_TOKEN")
            .expect("VERTEX_ACCESS_TOKEN should be present");

        println!(
            "token obtained (first 20 chars): {}...",
            &token[..20.min(token.len())]
        );

        // Step 2: call Vertex with the token.
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .build()
            .unwrap();

        let body = serde_json::json!({
            "anthropic_version": "vertex-2023-10-16",
            "max_tokens": 32,
            "messages": [{"role": "user", "content": "Say hello in one word."}]
        });

        let resp = client
            .post(&base_url)
            .bearer_auth(token)
            .header("anthropic-version", "2023-06-01")
            .json(&body)
            .send()
            .await
            .expect("request should succeed");

        let status = resp.status();
        let text = resp.text().await.unwrap_or_default();
        println!("status: {status}");
        println!("response: {text}");

        assert!(status.is_success(), "expected 2xx, got {status}: {text}");

        // Step 3: verify needs_refresh returns false with fresh token.
        creds.extend(updated);
        assert!(
            !refresher.needs_refresh(&creds),
            "should not need refresh immediately after obtaining token"
        );
    }
}
