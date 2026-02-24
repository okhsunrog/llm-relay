use tracing::{debug, error, info};

use super::LlmClient;
use super::error::LlmError;
use crate::convert::{thinking::build_thinking_params, to_openai};
use crate::types::anthropic::{Message, MessagesRequest, MessagesResponse};
use crate::types::common::{Provider, ThinkingConfig, ToolDefinition};
use crate::types::openai::{self, ChatRequest};

/// Options for a chat request.
#[derive(Default)]
pub struct ChatOptions<'a> {
    pub system: Option<&'a str>,
    pub tools: Option<&'a [ToolDefinition]>,
    pub thinking: Option<&'a ThinkingConfig>,
    pub temperature: Option<f32>,
}

impl LlmClient {
    /// Send a chat completion request using Anthropic message format.
    ///
    /// Automatically converts to OpenAI format if the provider is OpenAI-compatible.
    /// Always returns the response in Anthropic format (canonical).
    pub async fn chat(
        &self,
        messages: &[Message],
        options: ChatOptions<'_>,
    ) -> Result<MessagesResponse, LlmError> {
        info!(
            "Sending request to LLM (provider: {}, model: {}, messages: {})",
            self.config.provider,
            self.config.model,
            messages.len()
        );

        match self.config.provider {
            Provider::Anthropic => self.chat_anthropic(messages, &options).await,
            Provider::OpenAiCompatible => self.chat_openai_compat(messages, &options).await,
        }
    }

    /// Simple text-in, full-response-out call.
    ///
    /// Sends a single user message and returns the full response.
    /// Use `.text()` on the result to extract just the text content.
    pub async fn complete(
        &self,
        system: Option<&str>,
        user: &str,
        thinking: Option<&ThinkingConfig>,
    ) -> Result<MessagesResponse, LlmError> {
        let messages = vec![Message::user_text(user)];
        let options = ChatOptions {
            system,
            tools: None,
            thinking,
            temperature: None,
        };
        self.chat(&messages, options).await
    }

    /// Send a raw OpenAI-format chat request.
    ///
    /// Bypasses Anthropic format conversion — sends and receives OpenAI types directly.
    pub async fn chat_openai_raw(
        &self,
        request: &ChatRequest,
    ) -> Result<openai::ChatResponse, LlmError> {
        let url = format!("{}/v1/chat/completions", self.config.base_url);
        debug!("POST {url} (model: {})", request.model);

        let response = self
            .http
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.config.api_key))
            .header("content-type", "application/json")
            .json(request)
            .send()
            .await?;

        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            error!("API error {status}: {body}");
            return Err(LlmError::ApiError {
                status: status.as_u16(),
                body,
            });
        }

        let resp: openai::ChatResponse = response.json().await.map_err(|e| {
            error!("Failed to parse response: {e}");
            LlmError::ParseResponse(e.to_string())
        })?;

        Ok(resp)
    }

    // --- Private implementation ---

    async fn chat_anthropic(
        &self,
        messages: &[Message],
        options: &ChatOptions<'_>,
    ) -> Result<MessagesResponse, LlmError> {
        let (thinking, output_config) = build_thinking_params(options.thinking);

        let request_body = MessagesRequest {
            model: self.config.model.clone(),
            max_tokens: self.config.max_tokens,
            system: options.system.map(|s| s.to_string()),
            messages: messages.to_vec(),
            tools: options.tools.map(|t| t.to_vec()),
            thinking,
            output_config,
        };

        let url = format!("{}/v1/messages", self.config.base_url);
        debug!("POST {url} (model: {})", self.config.model);

        let response = self
            .http
            .post(&url)
            .header("x-api-key", &self.config.api_key)
            .header("anthropic-version", "2023-06-01")
            .header("content-type", "application/json")
            .json(&request_body)
            .send()
            .await?;

        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            error!("API error {status}: {body}");
            return Err(LlmError::ApiError {
                status: status.as_u16(),
                body,
            });
        }

        let resp: MessagesResponse = response.json().await.map_err(|e| {
            error!("Failed to parse response: {e}");
            LlmError::ParseResponse(e.to_string())
        })?;

        info!(
            "LLM responded (stop_reason: {}, content blocks: {})",
            resp.stop_reason,
            resp.content.len()
        );
        Ok(resp)
    }

    async fn chat_openai_compat(
        &self,
        messages: &[Message],
        options: &ChatOptions<'_>,
    ) -> Result<MessagesResponse, LlmError> {
        let openai_messages = to_openai::messages_to_openai(options.system, messages);

        let tools = options.tools.map(to_openai::tools_to_openai);

        let request_body = openai::ChatRequest {
            model: self.config.model.clone(),
            max_tokens: Some(self.config.max_tokens),
            messages: openai_messages,
            temperature: options.temperature,
            tools,
            response_format: None,
        };

        let url = format!("{}/v1/chat/completions", self.config.base_url);
        debug!("POST {url} (model: {})", self.config.model);

        let response = self
            .http
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.config.api_key))
            .header("content-type", "application/json")
            .json(&request_body)
            .send()
            .await?;

        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            error!("API error {status}: {body}");
            return Err(LlmError::ApiError {
                status: status.as_u16(),
                body,
            });
        }

        let openai_resp: openai::ChatResponse = response.json().await.map_err(|e| {
            error!("Failed to parse response: {e}");
            LlmError::ParseResponse(e.to_string())
        })?;

        let resp = to_openai::response_to_anthropic(openai_resp).map_err(LlmError::Conversion)?;

        info!(
            "LLM responded (stop_reason: {}, content blocks: {})",
            resp.stop_reason,
            resp.content.len()
        );
        Ok(resp)
    }
}
