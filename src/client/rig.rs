//! Adapters that let applications keep Rig's agent/tool orchestration while
//! using `llm-relay` as the provider configuration boundary.

use rig::providers::{anthropic, openai};

use super::{AuthScheme, ClientConfig, error::LlmError, normalized_api_base};

#[derive(Clone)]
pub enum RigClient {
    OpenAi(openai::CompletionsClient),
    Anthropic(anthropic::Client),
}

impl ClientConfig {
    /// Build the matching native Rig client with the same base URL, API key,
    /// and custom headers. OpenAI-compatible providers use Chat Completions,
    /// which has the broadest compatibility across local and hosted servers.
    pub fn rig_client(&self) -> Result<RigClient, LlmError> {
        match self.provider {
            crate::types::common::Provider::OpenAiCompatible => {
                Ok(RigClient::OpenAi(self.rig_openai_client()?))
            }
            crate::types::common::Provider::Anthropic => {
                Ok(RigClient::Anthropic(self.rig_anthropic_client()?))
            }
        }
    }

    pub fn rig_openai_client(&self) -> Result<openai::CompletionsClient, LlmError> {
        if self.provider != crate::types::common::Provider::OpenAiCompatible {
            return Err(LlmError::Config(
                "an OpenAI-compatible Rig client requires the OpenAI protocol".into(),
            ));
        }
        if !matches!(self.auth_scheme, AuthScheme::Bearer | AuthScheme::None) {
            return Err(LlmError::Config(
                "Rig's OpenAI adapter supports bearer or no-auth-compatible endpoints".into(),
            ));
        }
        // Rig's typed client requires a bearer credential. A harmless sentinel
        // keeps no-auth local servers compatible; such servers ignore the header.
        let api_key = if self.api_key.is_empty() {
            "llm-relay-local"
        } else {
            &self.api_key
        };
        let mut builder = openai::CompletionsClient::builder()
            .base_url(normalized_api_base(self))
            .api_key(api_key);
        builder = builder.http_headers(build_headers(&self.headers)?);
        builder
            .build()
            .map_err(|error| LlmError::Client(error.to_string()))
    }

    pub fn rig_anthropic_client(&self) -> Result<anthropic::Client, LlmError> {
        if self.provider != crate::types::common::Provider::Anthropic {
            return Err(LlmError::Config(
                "an Anthropic Rig client requires the Anthropic protocol".into(),
            ));
        }
        if !matches!(
            &self.auth_scheme,
            AuthScheme::Header(name) if name.eq_ignore_ascii_case("x-api-key")
        ) {
            return Err(LlmError::Config(
                "Rig's Anthropic adapter requires x-api-key authentication".into(),
            ));
        }
        let mut builder = anthropic::Client::builder()
            .base_url(normalized_api_base(self))
            .api_key(&self.api_key);
        builder = builder.http_headers(build_headers(&self.headers)?);
        builder
            .build()
            .map_err(|error| LlmError::Client(error.to_string()))
    }
}

fn build_headers(
    headers: &std::collections::BTreeMap<String, String>,
) -> Result<rig::http_client::HeaderMap, LlmError> {
    let mut destination = rig::http_client::HeaderMap::new();
    for (name, value) in headers {
        let name = name
            .parse::<http::HeaderName>()
            .map_err(|error| LlmError::Config(error.to_string()))?;
        let value = value
            .parse::<rig::http_client::HeaderValue>()
            .map_err(|error| LlmError::Config(error.to_string()))?;
        destination.insert(name, value);
    }
    Ok(destination)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rig_clients_keep_custom_api_bases() {
        let openai =
            ClientConfig::openai_compatible("http://localhost:11434/custom/v1", "", "qwen3")
                .rig_openai_client()
                .expect("OpenAI client");
        assert_eq!(openai.base_url(), "http://localhost:11434/custom/v1");

        let anthropic = ClientConfig::anthropic("key", "claude")
            .base_url("https://proxy.example/anthropic/v1/messages")
            .rig_anthropic_client()
            .expect("Anthropic client");
        assert_eq!(anthropic.base_url(), "https://proxy.example/anthropic");
    }
}
