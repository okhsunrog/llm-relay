use tracing::{debug, error, info};

use super::error::LlmError;
use super::LlmClient;
use crate::convert::{thinking::build_thinking_params, to_openai};
use crate::types::anthropic::{Message, MessagesRequest, MessagesResponse};
use crate::types::common::{Provider, ThinkingConfig, ToolDefinition};
use crate::types::openai::{self, ChatRequest};

/// Options for a chat request.
pub struct ChatOptions<'a> {
    pub system: Option<&'a str>,
    pub tools: Option<&'a [ToolDefinition]>,
    pub thinking: Option<&'a ThinkingConfig>,
}

impl Default for ChatOptions<'_> {
    fn default() -> Self {
        Self {
            system: None,
            tools: None,
            thinking: None,
        }
    }
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

    /// Simple text-in text-out call.
    ///
    /// Sends a single user message and returns the text response.
    pub async fn complete(
        &self,
        system: Option<&str>,
        user: &str,
        thinking: Option<&ThinkingConfig>,
    ) -> Result<String, LlmError> {
        let messages = vec![Message::user_text(user)];
        let options = ChatOptions {
            system,
            tools: None,
            thinking,
        };
        let resp = self.chat(&messages, options).await?;
        let text = resp.text();
        if text.is_empty() {
            Err(LlmError::EmptyResponse)
        } else {
            Ok(text)
        }
    }

    /// Send a raw OpenAI-format chat request.
    ///
    /// For projects that use OpenAI as their native format (chai-rs, fridge_tracker).
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

        let tools = options.tools.map(|t| to_openai::tools_to_openai(t));

        let request_body = openai::ChatRequest {
            model: self.config.model.clone(),
            max_tokens: Some(self.config.max_tokens),
            messages: openai_messages,
            temperature: None,
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

        let resp = to_openai::response_to_anthropic(openai_resp)
            .map_err(|e| LlmError::Conversion(e))?;

        info!(
            "LLM responded (stop_reason: {}, content blocks: {})",
            resp.stop_reason,
            resp.content.len()
        );
        Ok(resp)
    }
}
