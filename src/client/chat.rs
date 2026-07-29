use schemars::JsonSchema;
use serde::de::DeserializeOwned;
use serde_json::Value;
use tracing::{debug, info};

use super::LlmClient;
use super::error::LlmError;
use crate::convert::{thinking::build_thinking_params, to_openai};
use crate::types::anthropic::{
    Message, MessagesRequest, MessagesResponse, OutputConfig, OutputFormat,
};
use crate::types::common::{Provider, ResponseFormat, StopReason, ThinkingConfig, ToolDefinition};
use crate::types::openai::{self, ChatRequest};

/// Options for a chat request.
#[derive(Default)]
pub struct ChatOptions<'a> {
    pub system: Option<&'a str>,
    pub tools: Option<&'a [ToolDefinition]>,
    pub thinking: Option<&'a ThinkingConfig>,
    pub temperature: Option<f32>,
    pub response_format: Option<&'a ResponseFormat>,
    /// Require one named tool. The transport maps this to the provider's
    /// native tool-choice wire format.
    pub required_tool: Option<&'a str>,
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
    /// OpenAI-compatible providers use `response_format.json_schema`. Native
    /// Anthropic providers use `output_config.format` with constrained decoding.
    pub async fn complete_structured<T>(
        &self,
        user: &str,
        schema_name: &str,
        system: Option<&str>,
    ) -> Result<StructuredResponse<T>, LlmError>
    where
        T: DeserializeOwned + JsonSchema,
    {
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
        let response = match self.config.provider {
            Provider::OpenAiCompatible => {
                let response_format = ResponseFormat::json_schema(schema_name, schema, true);
                self.complete(
                    user,
                    ChatOptions {
                        system,
                        temperature: Some(0.0),
                        response_format: Some(&response_format),
                        ..ChatOptions::default()
                    },
                )
                .await?
            }
            Provider::Anthropic => {
                let schema = prepare_anthropic_schema(schema);
                let response_format = ResponseFormat::json_schema(schema_name, schema, true);
                self.complete(
                    user,
                    ChatOptions {
                        system,
                        temperature: Some(0.0),
                        response_format: Some(&response_format),
                        ..ChatOptions::default()
                    },
                )
                .await?
            }
        };
        if matches!(response.stop_reason, StopReason::MaxTokens) {
            return Err(LlmError::InvalidStructuredOutput {
                error: "provider stopped at max_tokens".into(),
                body: response.text().chars().take(4_096).collect(),
            });
        }
        if matches!(&response.stop_reason, StopReason::Other(reason) if reason == "refusal") {
            return Err(LlmError::InvalidStructuredOutput {
                error: "provider refused the structured request".into(),
                body: response.text().chars().take(4_096).collect(),
            });
        }
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
        let (thinking, mut output_config) = build_thinking_params(options.thinking);
        if let Some(response_format) = options.response_format {
            let ResponseFormat::JsonSchema { json_schema } = response_format else {
                return Err(LlmError::Config(
                    "native Anthropic structured output requires a JSON Schema".into(),
                ));
            };
            output_config
                .get_or_insert_with(OutputConfig::default)
                .format = Some(OutputFormat::JsonSchema {
                schema: json_schema.schema.clone(),
            });
        }

        let request_body = MessagesRequest {
            model: self.config.model.clone(),
            max_tokens: self.config.max_tokens,
            system: options.system.map(|s| s.to_string()),
            messages: messages.to_vec(),
            tools: options.tools.map(|t| t.to_vec()),
            thinking,
            output_config,
            tool_choice: options
                .required_tool
                .map(|name| serde_json::json!({"type": "tool", "name": name})),
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
            tool_choice: options
                .required_tool
                .map(|name| serde_json::json!({"type": "function", "function": {"name": name}})),
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

fn prepare_anthropic_schema(mut schema: Value) -> Value {
    normalize_anthropic_schema(&mut schema);
    schema
}

fn normalize_anthropic_schema(schema: &mut Value) {
    match schema {
        Value::Array(items) => {
            for item in items {
                normalize_anthropic_schema(item);
            }
        }
        Value::Object(object) => {
            object.remove("$schema");
            object.remove("default");
            object.remove("examples");
            object.remove("minimum");
            object.remove("maximum");
            object.remove("exclusiveMinimum");
            object.remove("exclusiveMaximum");
            object.remove("multipleOf");
            object.remove("minLength");
            object.remove("maxLength");
            object.remove("minItems");
            object.remove("maxItems");
            object.remove("minProperties");
            object.remove("maxProperties");
            object.remove("format");

            let property_names = object
                .get("properties")
                .and_then(Value::as_object)
                .map(|properties| properties.keys().cloned().collect::<Vec<_>>());
            if let Some(property_names) = property_names {
                object.insert("additionalProperties".into(), Value::Bool(false));
                object.insert(
                    "required".into(),
                    Value::Array(property_names.into_iter().map(Value::String).collect()),
                );
            }
            for value in object.values_mut() {
                normalize_anthropic_schema(value);
            }
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::prepare_anthropic_schema;

    #[test]
    fn anthropic_schema_is_strict_and_uses_only_supported_constraints() {
        let schema = json!({
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "properties": {
                "score": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 100,
                    "default": 50
                },
                "tags": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "minLength": 1
                    }
                },
                "metadata": {
                    "type": "object",
                    "properties": {
                        "created_at": {
                            "type": "string",
                            "format": "date-time"
                        }
                    }
                }
            }
        });

        let prepared = prepare_anthropic_schema(schema);

        assert_eq!(prepared["required"], json!(["metadata", "score", "tags"]));
        assert_eq!(prepared["additionalProperties"], false);
        assert_eq!(
            prepared["properties"]["metadata"]["required"],
            json!(["created_at"])
        );
        assert_eq!(
            prepared["properties"]["metadata"]["additionalProperties"],
            false
        );
        assert!(prepared.get("$schema").is_none());
        assert!(
            prepared["properties"]["score"]
                .as_object()
                .is_some_and(|score| !score.contains_key("minimum")
                    && !score.contains_key("maximum")
                    && !score.contains_key("default"))
        );
        assert!(
            prepared["properties"]["tags"]["items"]
                .as_object()
                .is_some_and(|items| !items.contains_key("minLength"))
        );
        assert!(
            prepared["properties"]["metadata"]["properties"]["created_at"]
                .as_object()
                .is_some_and(|created_at| !created_at.contains_key("format"))
        );
    }
}
