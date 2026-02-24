use std::time::Duration;

use tracing::{debug, error};

use super::error::LlmError;
use crate::types::openai::{EmbeddingsRequest, EmbeddingsResponse};

/// Configuration for the embeddings client.
#[derive(Debug, Clone)]
pub struct EmbeddingsConfig {
    pub base_url: String,
    pub api_key: String,
    pub model: String,
    pub timeout: Duration,
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
        }
    }

    /// Create config for OpenRouter embeddings.
    pub fn openrouter(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        Self::openai_compatible("https://openrouter.ai/api/v1", api_key, model)
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
}

/// Embeddings client (OpenAI-compatible API only).
pub struct EmbeddingsClient {
    http: reqwest::Client,
    config: EmbeddingsConfig,
}

impl EmbeddingsClient {
    pub fn new(config: EmbeddingsConfig) -> Result<Self, LlmError> {
        let http = reqwest::Client::builder()
            .timeout(config.timeout)
            .build()
            .map_err(|e| LlmError::Client(e.to_string()))?;
        Ok(Self { http, config })
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
        };

        let url = format!("{}/embeddings", self.config.base_url);
        debug!(
            "POST {url} (model: {}, count: {expected_count})",
            self.config.model
        );

        let response = self
            .http
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.config.api_key))
            .header("content-type", "application/json")
            .json(&request)
            .send()
            .await?;

        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            error!("Embeddings API error {status}: {body}");
            return Err(LlmError::ApiError {
                status: status.as_u16(),
                body,
            });
        }

        let resp: EmbeddingsResponse = response.json().await.map_err(|e| {
            error!("Failed to parse embeddings response: {e}");
            LlmError::ParseResponse(e.to_string())
        })?;

        if resp.data.len() != expected_count {
            return Err(LlmError::ParseResponse(format!(
                "Expected {} embeddings, got {}",
                expected_count,
                resp.data.len()
            )));
        }

        // Sort by index to ensure correct order
        let mut data = resp.data;
        data.sort_by_key(|e| e.index);

        Ok(data.into_iter().map(|e| e.embedding).collect())
    }

    /// Create embedding for a single text.
    pub async fn create_embedding(&self, text: &str) -> Result<Vec<f32>, LlmError> {
        let mut results = self.create_embeddings(&[text]).await?;
        results.pop().ok_or(LlmError::EmptyResponse)
    }
}
