use std::time::Duration;

use tracing::debug;

use super::{AuthScheme, ClientConfig, LlmClient, RetryPolicy, error::LlmError};
use crate::types::openai::{EmbeddingsRequest, EmbeddingsResponse};

/// Configuration for the embeddings client.
#[derive(Debug, Clone)]
pub struct EmbeddingsConfig {
    pub base_url: String,
    pub api_key: String,
    pub model: String,
    pub timeout: Duration,
    pub dimensions: Option<u32>,
    pub input_type: Option<String>,
    pub encoding_format: Option<String>,
    pub auth_scheme: AuthScheme,
    pub headers: std::collections::BTreeMap<String, String>,
    pub retry_policy: RetryPolicy,
    pub max_response_bytes: usize,
}

impl EmbeddingsConfig {
    /// Create config for any OpenAI-compatible embeddings API.
    pub fn openai_compatible(
        base_url: impl Into<String>,
        api_key: impl Into<String>,
        model: impl Into<String>,
    ) -> Self {
        Self {
            base_url: base_url.into(),
            api_key: api_key.into(),
            model: model.into(),
            timeout: Duration::from_secs(120),
            dimensions: None,
            input_type: None,
            encoding_format: None,
            auth_scheme: AuthScheme::Bearer,
            headers: std::collections::BTreeMap::new(),
            retry_policy: RetryPolicy::default(),
            max_response_bytes: 64 * 1024 * 1024,
        }
    }

    /// Create config for OpenRouter embeddings.
    pub fn openrouter(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        Self::openai_compatible("https://openrouter.ai/api/v1", api_key, model)
    }

    pub fn openai(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        Self::openai_compatible("https://api.openai.com/v1", api_key, model)
    }

    #[must_use]
    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    #[must_use]
    pub fn base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
    }

    #[must_use]
    pub fn dimensions(mut self, dimensions: u32) -> Self {
        self.dimensions = Some(dimensions);
        self
    }

    #[must_use]
    pub fn input_type(mut self, input_type: impl Into<String>) -> Self {
        self.input_type = Some(input_type.into());
        self
    }

    #[must_use]
    pub fn encoding_format(mut self, encoding_format: impl Into<String>) -> Self {
        self.encoding_format = Some(encoding_format.into());
        self
    }

    #[must_use]
    pub fn header(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.headers.insert(name.into(), value.into());
        self
    }

    #[must_use]
    pub fn without_auth(mut self) -> Self {
        self.auth_scheme = AuthScheme::None;
        self
    }
}

/// Embeddings client (OpenAI-compatible API only).
pub struct EmbeddingsClient {
    inner: LlmClient,
    config: EmbeddingsConfig,
}

impl EmbeddingsClient {
    pub fn new(config: EmbeddingsConfig) -> Result<Self, LlmError> {
        let mut client_config =
            ClientConfig::openai_compatible(&config.base_url, &config.api_key, &config.model)
                .timeout(config.timeout)
                .auth_scheme(config.auth_scheme.clone())
                .retry_policy(config.retry_policy.clone())
                .max_response_bytes(config.max_response_bytes);
        client_config.headers = config.headers.clone();
        Ok(Self {
            inner: LlmClient::new(client_config)?,
            config,
        })
    }

    /// Create embeddings for multiple texts (batch).
    ///
    /// Returns vectors sorted by input order.
    pub async fn create_embeddings(
        &self,
        texts: &[impl AsRef<str>],
    ) -> Result<Vec<Vec<f32>>, LlmError> {
        let expected_count = texts.len();
        let input: Vec<String> = texts.iter().map(|t| t.as_ref().to_string()).collect();

        let request = EmbeddingsRequest {
            model: self.config.model.clone(),
            input,
            dimensions: self.config.dimensions,
            input_type: self.config.input_type.clone(),
            encoding_format: self.config.encoding_format.clone(),
        };

        let url = self.inner.endpoint("embeddings");
        debug!(
            "POST {url} (model: {}, count: {expected_count})",
            self.config.model
        );

        let body = self.inner.send_json(&url, &request).await?;
        let resp: EmbeddingsResponse = serde_json::from_slice(&body)
            .map_err(|error| LlmError::ParseResponse(error.to_string()))?;

        if resp.data.len() != expected_count {
            return Err(LlmError::ParseResponse(format!(
                "Expected {} embeddings, got {}",
                expected_count,
                resp.data.len()
            )));
        }

        // Sort by index to ensure correct order
        let mut data = resp.data;
        let mut seen = std::collections::HashSet::new();
        if data
            .iter()
            .any(|embedding| embedding.index >= expected_count || !seen.insert(embedding.index))
        {
            return Err(LlmError::ParseResponse(
                "Embedding response contained an invalid or duplicate index".into(),
            ));
        }
        data.sort_by_key(|e| e.index);

        Ok(data.into_iter().map(|e| e.embedding).collect())
    }

    /// Create embedding for a single text.
    pub async fn create_embedding(&self, text: &str) -> Result<Vec<f32>, LlmError> {
        let mut results = self.create_embeddings(&[text]).await?;
        results.pop().ok_or(LlmError::EmptyResponse)
    }
}
