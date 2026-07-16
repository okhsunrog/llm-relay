use std::collections::BTreeMap;
use std::time::Duration;

use crate::types::common::Provider;

pub mod chat;
#[cfg(feature = "embeddings")]
pub mod embeddings;
pub mod error;
#[cfg(feature = "rig")]
pub mod rig;
#[cfg(feature = "streaming")]
pub mod streaming;

pub use chat::ChatOptions;
#[cfg(feature = "embeddings")]
pub use embeddings::{EmbeddingsClient, EmbeddingsConfig};
pub use error::LlmError;
#[cfg(feature = "streaming")]
pub use streaming::{ChatStream, StreamEvent};

const DEFAULT_MAX_RESPONSE_BYTES: usize = 16 * 1024 * 1024;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AuthScheme {
    Bearer,
    Header(String),
    None,
}

#[derive(Debug, Clone)]
pub struct RetryPolicy {
    pub max_retries: u32,
    pub initial_backoff: Duration,
    pub max_backoff: Duration,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_retries: 2,
            initial_backoff: Duration::from_millis(250),
            max_backoff: Duration::from_secs(4),
        }
    }
}

/// Configuration for the LLM client.
#[derive(Debug, Clone)]
pub struct ClientConfig {
    pub provider: Provider,
    pub base_url: String,
    pub api_key: String,
    pub timeout: Duration,
    pub model: String,
    pub max_tokens: u32,
    pub auth_scheme: AuthScheme,
    pub headers: BTreeMap<String, String>,
    pub retry_policy: RetryPolicy,
    pub max_response_bytes: usize,
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
            auth_scheme: AuthScheme::Header("x-api-key".into()),
            headers: BTreeMap::new(),
            retry_policy: RetryPolicy::default(),
            max_response_bytes: DEFAULT_MAX_RESPONSE_BYTES,
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
            auth_scheme: AuthScheme::Bearer,
            headers: BTreeMap::new(),
            retry_policy: RetryPolicy::default(),
            max_response_bytes: DEFAULT_MAX_RESPONSE_BYTES,
        }
    }

    /// Create config for the official OpenAI API.
    pub fn openai(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        Self::openai_compatible("https://api.openai.com/v1", api_key, model)
    }

    /// Create config for OpenRouter's OpenAI-compatible API.
    pub fn openrouter(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        Self::openai_compatible("https://openrouter.ai/api/v1", api_key, model)
    }

    /// Create config for a no-auth local OpenAI-compatible server. `base_url`
    /// is the complete API base, commonly `http://localhost:11434/v1`.
    pub fn local_openai_compatible(base_url: impl Into<String>, model: impl Into<String>) -> Self {
        Self::openai_compatible(base_url, "", model).without_auth()
    }

    #[must_use]
    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    #[must_use]
    pub fn max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    #[must_use]
    pub fn base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
    }

    #[must_use]
    pub fn header(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.headers.insert(name.into(), value.into());
        self
    }

    #[must_use]
    pub fn auth_scheme(mut self, auth_scheme: AuthScheme) -> Self {
        self.auth_scheme = auth_scheme;
        self
    }

    #[must_use]
    pub fn without_auth(mut self) -> Self {
        self.auth_scheme = AuthScheme::None;
        self
    }

    #[must_use]
    pub fn retry_policy(mut self, retry_policy: RetryPolicy) -> Self {
        self.retry_policy = retry_policy;
        self
    }

    #[must_use]
    pub fn max_response_bytes(mut self, bytes: usize) -> Self {
        self.max_response_bytes = bytes.max(1);
        self
    }
}

/// The main LLM client.
#[derive(Clone)]
pub struct LlmClient {
    pub(crate) http: reqwest::Client,
    pub(crate) config: ClientConfig,
}

impl LlmClient {
    pub fn new(config: ClientConfig) -> Result<Self, LlmError> {
        validate_base_url(&config.base_url)?;
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

    pub(crate) fn endpoint(&self, path: &str) -> String {
        format!(
            "{}/{}",
            normalized_api_base(&self.config),
            path.trim_start_matches('/')
        )
    }

    pub(crate) fn request(&self, url: &str) -> Result<reqwest::RequestBuilder, LlmError> {
        let mut request = self
            .http
            .post(url)
            .header("content-type", "application/json");
        request = match &self.config.auth_scheme {
            AuthScheme::Bearer if !self.config.api_key.is_empty() => {
                request.bearer_auth(&self.config.api_key)
            }
            AuthScheme::Header(name) if !self.config.api_key.is_empty() => {
                let name = reqwest::header::HeaderName::from_bytes(name.as_bytes())
                    .map_err(|error| LlmError::Config(error.to_string()))?;
                request.header(name, &self.config.api_key)
            }
            _ => request,
        };
        for (name, value) in &self.config.headers {
            let name = reqwest::header::HeaderName::from_bytes(name.as_bytes())
                .map_err(|error| LlmError::Config(error.to_string()))?;
            let value = reqwest::header::HeaderValue::from_str(value)
                .map_err(|error| LlmError::Config(error.to_string()))?;
            request = request.header(name, value);
        }
        if self.config.provider == Provider::Anthropic
            && !self
                .config
                .headers
                .keys()
                .any(|name| name.eq_ignore_ascii_case("anthropic-version"))
        {
            request = request.header("anthropic-version", "2023-06-01");
        }
        Ok(request)
    }

    pub(crate) async fn send_json<T: serde::Serialize + ?Sized>(
        &self,
        url: &str,
        body: &T,
    ) -> Result<Vec<u8>, LlmError> {
        let body = serde_json::to_vec(body).map_err(|error| LlmError::Client(error.to_string()))?;
        let mut attempt = 0;
        loop {
            let response = self.request(url)?.body(body.clone()).send().await?;
            let status = response.status();
            let retryable =
                status.as_u16() == 408 || status.as_u16() == 429 || status.is_server_error();
            if retryable && attempt < self.config.retry_policy.max_retries {
                let shift = attempt.min(16);
                let factor = 1u32 << shift;
                let delay = self
                    .config
                    .retry_policy
                    .initial_backoff
                    .saturating_mul(factor)
                    .min(self.config.retry_policy.max_backoff);
                attempt += 1;
                tracing::warn!(%status, attempt, ?delay, "retrying LLM request");
                tokio::time::sleep(delay).await;
                continue;
            }
            let bytes = response.bytes().await?;
            if bytes.len() > self.config.max_response_bytes {
                return Err(LlmError::ResponseTooLarge {
                    limit: self.config.max_response_bytes,
                    actual: bytes.len(),
                });
            }
            if !status.is_success() {
                return Err(LlmError::ApiError {
                    status: status.as_u16(),
                    body: String::from_utf8_lossy(&bytes).into_owned(),
                });
            }
            return Ok(bytes.to_vec());
        }
    }
}

pub(crate) fn normalized_api_base(config: &ClientConfig) -> String {
    let base = config.base_url.trim().trim_end_matches('/');
    match config.provider {
        Provider::OpenAiCompatible => base
            .strip_suffix("/chat/completions")
            .or_else(|| base.strip_suffix("/embeddings"))
            .unwrap_or(base)
            .trim_end_matches('/')
            .to_string(),
        Provider::Anthropic => base
            .strip_suffix("/v1/messages")
            .or_else(|| base.strip_suffix("/messages"))
            .or_else(|| base.strip_suffix("/v1"))
            .unwrap_or(base)
            .trim_end_matches('/')
            .to_string(),
    }
}

fn validate_base_url(value: &str) -> Result<(), LlmError> {
    let url =
        reqwest::Url::parse(value.trim()).map_err(|error| LlmError::Config(error.to_string()))?;
    if !matches!(url.scheme(), "http" | "https") || url.host_str().is_none() {
        return Err(LlmError::Config(
            "base URL must use http or https and contain a host".into(),
        ));
    }
    if url.query().is_some() || url.fragment().is_some() {
        return Err(LlmError::Config(
            "base URL cannot contain a query string or fragment".into(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn endpoint_contract_normalizes_pasted_provider_urls() {
        let openai = ClientConfig::openai_compatible(
            "https://proxy.example/v1/chat/completions/",
            "key",
            "model",
        );
        assert_eq!(normalized_api_base(&openai), "https://proxy.example/v1");

        let anthropic = ClientConfig::anthropic("key", "model")
            .base_url("https://proxy.example/anthropic/v1/messages");
        assert_eq!(
            normalized_api_base(&anthropic),
            "https://proxy.example/anthropic"
        );
    }

    #[test]
    fn rejects_unsafe_or_ambiguous_base_urls() {
        assert!(
            LlmClient::new(ClientConfig::openai_compatible(
                "file:///tmp/socket",
                "",
                "model"
            ))
            .is_err()
        );
        assert!(
            LlmClient::new(ClientConfig::openai_compatible(
                "https://proxy.example/v1?key=secret",
                "",
                "model"
            ))
            .is_err()
        );
    }
}
