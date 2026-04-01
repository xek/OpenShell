// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::{DiscoveredProvider, ProviderError, ProviderPlugin};
use std::collections::HashMap;

pub struct VertexProvider;

impl ProviderPlugin for VertexProvider {
    fn id(&self) -> &'static str {
        "vertex"
    }

    /// Discover Vertex AI credentials from the environment.
    ///
    /// Sources checked in priority order:
    /// 1. `GOOGLE_SERVICE_ACCOUNT_JSON` — inline service account JSON
    /// 2. `GOOGLE_APPLICATION_CREDENTIALS` — path to a credential JSON file
    ///    (service account or ADC user credential)
    /// 3. Application Default Credentials well-known file
    ///    (`~/.config/gcloud/application_default_credentials.json`)
    ///
    /// The gateway's credential refresh loop exchanges the stored credential
    /// for a short-lived OAuth2 Bearer token before inference calls are made.
    fn discover_existing(&self) -> Result<Option<DiscoveredProvider>, ProviderError> {
        let mut credentials = HashMap::new();
        let mut config = HashMap::new();

        // Credential discovery (in priority order).
        if let Ok(json) = std::env::var("GOOGLE_SERVICE_ACCOUNT_JSON") {
            if !json.trim().is_empty() {
                credentials.insert("GOOGLE_SERVICE_ACCOUNT_JSON".to_string(), json);
            }
        } else if let Ok(path) = std::env::var("GOOGLE_APPLICATION_CREDENTIALS") {
            if let Ok(json) = std::fs::read_to_string(&path) {
                credentials.insert("GOOGLE_SERVICE_ACCOUNT_JSON".to_string(), json);
            }
        } else if let Ok(home) = std::env::var("HOME") {
            let adc_path = format!("{home}/.config/gcloud/application_default_credentials.json");
            if let Ok(json) = std::fs::read_to_string(&adc_path) {
                credentials.insert("GOOGLE_SERVICE_ACCOUNT_JSON".to_string(), json);
            }
        }

        if credentials.is_empty() {
            return Ok(None);
        }

        // Project ID (checked in priority order across common env var names).
        for key in &[
            "GOOGLE_CLOUD_PROJECT",
            "VERTEXAI_PROJECT",
            "ANTHROPIC_VERTEX_PROJECT_ID",
            "GCLOUD_PROJECT",
        ] {
            if let Ok(v) = std::env::var(key) {
                if !v.trim().is_empty() {
                    config.insert("VERTEX_PROJECT".to_string(), v);
                    break;
                }
            }
        }

        // Location / region.
        for key in &[
            "VERTEXAI_LOCATION",
            "VERTEX_LOCATION",
            "GOOGLE_CLOUD_REGION",
        ] {
            if let Ok(v) = std::env::var(key) {
                if !v.trim().is_empty() {
                    config.insert("VERTEX_LOCATION".to_string(), v);
                    break;
                }
            }
        }

        // ANTHROPIC_BASE_URL routes Claude Code through the cluster inference proxy
        // (inference.local), which injects the Vertex Bearer token on outbound calls.
        credentials.insert(
            "ANTHROPIC_BASE_URL".to_string(),
            "https://inference.local".to_string(),
        );

        Ok(Some(DiscoveredProvider {
            credentials,
            config,
        }))
    }

    fn credential_env_vars(&self) -> &'static [&'static str] {
        &["GOOGLE_SERVICE_ACCOUNT_JSON"]
    }
}

#[cfg(test)]
mod tests {
    use super::VertexProvider;
    use crate::ProviderPlugin;

    #[test]
    fn vertex_provider_id() {
        assert_eq!(VertexProvider.id(), "vertex");
    }

    #[test]
    fn vertex_provider_no_env_returns_none() {
        // Remove env vars if set (best-effort in test environment).
        unsafe {
            std::env::remove_var("GOOGLE_SERVICE_ACCOUNT_JSON");
            std::env::remove_var("GOOGLE_APPLICATION_CREDENTIALS");
        }
        let result = VertexProvider.discover_existing().expect("discovery");
        assert!(result.is_none());
    }
}
