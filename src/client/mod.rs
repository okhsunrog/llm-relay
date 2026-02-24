use std::time::Duration;

use crate::types::common::Provider;

pub mod chat;
#[cfg(feature = "embeddings")]
pub mod embeddings;
pub mod error;

pub use chat::ChatOptions;
#[cfg(feature = "embeddings")]
pub use embeddings::{EmbeddingsClient, EmbeddingsConfig};
pub use error::LlmError;

/// Configuration for the LLM client.
#[derive(Debug, Clone)]
pub struct ClientConfig {
    pub provider: Provider,
    pub base_url: String,
    pub api_key: String,
    pub timeout: Duration,
    pub model: String,
    pub max_tokens: u32,
}

impl ClientConfig {
    /// Create config for the Anthropic API.
    pub fn anthropic(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            provider: Provider::Anthropic,
            base_url: "https://api.anthropic.com".to_string(),
            api_key: api_key.into(),
            timeout: Duration::from_secs(180),
            model: model.into(),
            max_tokens: 16384,
        }
    }

    /// Create config for an OpenAI-compatible API (OpenRouter, OpenAI, Ollama, etc.).
    pub fn openai_compatible(
        base_url: impl Into<String>,
        api_key: impl Into<String>,
        model: impl Into<String>,
    ) -> Self {
        Self {
            provider: Provider::OpenAiCompatible,
            base_url: base_url.into(),
            api_key: api_key.into(),
            timeout: Duration::from_secs(60),
            model: model.into(),
            max_tokens: 16384,
        }
    }

    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    pub fn max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    pub fn base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
    }
}

/// The main LLM client.
pub struct LlmClient {
    pub(crate) http: reqwest::Client,
    pub(crate) config: ClientConfig,
}

impl LlmClient {
    pub fn new(config: ClientConfig) -> Result<Self, LlmError> {
        let http = reqwest::Client::builder()
            .timeout(config.timeout)
            .build()
            .map_err(|e| LlmError::Client(e.to_string()))?;
        Ok(Self { http, config })
    }

    /// Get a reference to the client config.
    pub fn config(&self) -> &ClientConfig {
        &self.config
    }
}
