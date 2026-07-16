use schemars::JsonSchema;
use serde::de::DeserializeOwned;
use tracing::{debug, info};

use super::LlmClient;
use super::error::LlmError;
use crate::convert::{thinking::build_thinking_params, to_openai};
use crate::types::anthropic::{Message, MessagesRequest, MessagesResponse};
use crate::types::common::{Provider, ResponseFormat, ThinkingConfig, ToolDefinition};
use crate::types::openai::{self, ChatRequest};

/// Options for a chat request.
#[derive(Default)]
pub struct ChatOptions<'a> {
    pub system: Option<&'a str>,
    pub tools: Option<&'a [ToolDefinition]>,
    pub thinking: Option<&'a ThinkingConfig>,
    pub temperature: Option<f32>,
    pub response_format: Option<&'a ResponseFormat>,
}

#[derive(Debug, Clone)]
pub struct StructuredResponse<T> {
    pub data: T,
    pub usage: crate::types::common::Usage,
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
        user: &str,
        options: ChatOptions<'_>,
    ) -> Result<MessagesResponse, LlmError> {
        let messages = vec![Message::user_text(user)];
        self.chat(&messages, options).await
    }

    /// Complete a request with a strict JSON Schema response and deserialize it.
    ///
    /// This uses the OpenAI-compatible `response_format.json_schema` contract.
    /// Native Anthropic Messages clients do not share that wire contract and
    /// are rejected instead of silently falling back to prompt-only JSON.
    pub async fn complete_structured<T>(
        &self,
        user: &str,
        schema_name: &str,
        system: Option<&str>,
    ) -> Result<StructuredResponse<T>, LlmError>
    where
        T: DeserializeOwned + JsonSchema,
    {
        if self.config.provider != Provider::OpenAiCompatible {
            return Err(LlmError::Config(
                "strict JSON Schema output requires an OpenAI-compatible provider".into(),
            ));
        }
        if schema_name.is_empty()
            || schema_name.len() > 64
            || !schema_name.chars().all(|character| {
                character.is_ascii_alphanumeric() || matches!(character, '_' | '-')
            })
        {
            return Err(LlmError::Config(
                "schema name must contain 1 to 64 ASCII letters, digits, underscores, or hyphens"
                    .into(),
            ));
        }

        let schema = serde_json::to_value(schemars::schema_for!(T))
            .map_err(|error| LlmError::Config(error.to_string()))?;
        let response_format = ResponseFormat::json_schema(schema_name, schema, true);
        let response = self
            .complete(
                user,
                ChatOptions {
                    system,
                    temperature: Some(0.0),
                    response_format: Some(&response_format),
                    ..ChatOptions::default()
                },
            )
            .await?;
        let text = response.text();
        if text.trim().is_empty() {
            return Err(LlmError::EmptyResponse);
        }
        let data =
            serde_json::from_str(&text).map_err(|error| LlmError::InvalidStructuredOutput {
                error: error.to_string(),
                body: text.chars().take(4_096).collect(),
            })?;
        Ok(StructuredResponse {
            data,
            usage: response.usage.unwrap_or_default(),
        })
    }

    /// Send a raw OpenAI-format chat request.
    ///
    /// Bypasses Anthropic format conversion — sends and receives OpenAI types directly.
    pub async fn chat_openai_raw(
        &self,
        request: &ChatRequest,
    ) -> Result<openai::ChatResponse, LlmError> {
        let url = self.endpoint("chat/completions");
        debug!("POST {url} (model: {})", request.model);

        let body = self.send_json(&url, request).await?;
        let resp: openai::ChatResponse = serde_json::from_slice(&body)
            .map_err(|error| LlmError::ParseResponse(error.to_string()))?;

        Ok(resp)
    }

    // --- Private implementation ---

    async fn chat_anthropic(
        &self,
        messages: &[Message],
        options: &ChatOptions<'_>,
    ) -> Result<MessagesResponse, LlmError> {
        if options.response_format.is_some() {
            return Err(LlmError::Config(
                "response_format is not supported by the native Anthropic Messages transport"
                    .into(),
            ));
        }
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

        let url = self.endpoint("v1/messages");
        debug!("POST {url} (model: {})", self.config.model);

        let body = self.send_json(&url, &request_body).await?;
        let resp: MessagesResponse = serde_json::from_slice(&body)
            .map_err(|error| LlmError::ParseResponse(error.to_string()))?;

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
            response_format: options.response_format.cloned(),
        };

        let url = self.endpoint("chat/completions");
        debug!("POST {url} (model: {})", self.config.model);

        let body = self.send_json(&url, &request_body).await?;
        let openai_resp: openai::ChatResponse = serde_json::from_slice(&body)
            .map_err(|error| LlmError::ParseResponse(error.to_string()))?;

        let resp = to_openai::response_to_anthropic(openai_resp).map_err(LlmError::Conversion)?;

        info!(
            "LLM responded (stop_reason: {}, content blocks: {})",
            resp.stop_reason,
            resp.content.len()
        );
        Ok(resp)
    }
}
