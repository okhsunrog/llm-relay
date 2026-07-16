use std::pin::Pin;

use eventsource_stream::Eventsource;
use futures_core::Stream;
use futures_util::{StreamExt, stream};
use serde::{Deserialize, Serialize};

use super::{ChatOptions, LlmClient, error::LlmError};
use crate::convert::{thinking::build_thinking_params, to_openai};
use crate::types::anthropic::{Message, MessagesRequest};
use crate::types::common::{Provider, Usage};
use crate::types::openai::ChatRequest;

pub type ChatStream = Pin<Box<dyn Stream<Item = Result<StreamEvent, LlmError>> + Send>>;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum StreamEvent {
    TextDelta {
        text: String,
    },
    ThinkingDelta {
        text: String,
    },
    ToolCallDelta {
        index: usize,
        id: Option<String>,
        name: Option<String>,
        arguments: String,
    },
    ToolCallComplete {
        index: usize,
    },
    Usage {
        usage: Usage,
    },
    Done {
        stop_reason: Option<String>,
    },
}

impl LlmClient {
    /// Stream a chat completion and normalize provider SSE events.
    ///
    /// The returned stream intentionally exposes tool argument deltas instead
    /// of buffering them. Callers that execute tools should accumulate deltas
    /// by `index`, then parse the completed JSON on `ToolCallComplete`/`Done`.
    pub async fn chat_stream(
        &self,
        messages: &[Message],
        options: ChatOptions<'_>,
    ) -> Result<ChatStream, LlmError> {
        let (url, body) = match self.config.provider {
            Provider::OpenAiCompatible => {
                let request = ChatRequest {
                    model: self.config.model.clone(),
                    max_tokens: Some(self.config.max_tokens),
                    messages: to_openai::messages_to_openai(options.system, messages),
                    temperature: options.temperature,
                    tools: options.tools.map(to_openai::tools_to_openai),
                    response_format: options.response_format.cloned(),
                };
                let mut value = serde_json::to_value(request)
                    .map_err(|error| LlmError::Client(error.to_string()))?;
                value["stream"] = serde_json::json!(true);
                value["stream_options"] = serde_json::json!({"include_usage": true});
                (self.endpoint("chat/completions"), value)
            }
            Provider::Anthropic => {
                let (thinking, output_config) = build_thinking_params(options.thinking);
                let request = MessagesRequest {
                    model: self.config.model.clone(),
                    max_tokens: self.config.max_tokens,
                    system: options.system.map(str::to_string),
                    messages: messages.to_vec(),
                    tools: options.tools.map(<[_]>::to_vec),
                    thinking,
                    output_config,
                };
                let mut value = serde_json::to_value(request)
                    .map_err(|error| LlmError::Client(error.to_string()))?;
                value["stream"] = serde_json::json!(true);
                (self.endpoint("v1/messages"), value)
            }
        };

        let response = self.request(&url)?.json(&body).send().await?;
        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            return Err(LlmError::ApiError {
                status: status.as_u16(),
                body,
            });
        }

        let provider = self.config.provider;
        let events = response
            .bytes_stream()
            .eventsource()
            .map(move |event| match event {
                Ok(event) => match provider {
                    Provider::OpenAiCompatible => parse_openai_event(&event.data),
                    Provider::Anthropic => parse_anthropic_event(&event.event, &event.data),
                },
                Err(error) => vec![Err(LlmError::Stream(error.to_string()))],
            })
            .flat_map(stream::iter);
        Ok(Box::pin(events))
    }
}

fn parse_openai_event(data: &str) -> Vec<Result<StreamEvent, LlmError>> {
    if data.trim() == "[DONE]" {
        return vec![Ok(StreamEvent::Done { stop_reason: None })];
    }
    let value: serde_json::Value = match serde_json::from_str(data) {
        Ok(value) => value,
        Err(error) => return vec![Err(LlmError::Stream(error.to_string()))],
    };
    let mut events = Vec::new();
    if let Some(usage) = value.get("usage").filter(|usage| !usage.is_null()) {
        let input_tokens = usage
            .get("prompt_tokens")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or_default();
        let output_tokens = usage
            .get("completion_tokens")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or_default();
        events.push(Ok(StreamEvent::Usage {
            usage: Usage {
                input_tokens,
                output_tokens,
                cache_creation_input_tokens: usage
                    .pointer("/prompt_tokens_details/cache_write_tokens")
                    .and_then(serde_json::Value::as_u64),
                cache_read_input_tokens: usage
                    .pointer("/prompt_tokens_details/cached_tokens")
                    .and_then(serde_json::Value::as_u64),
                reasoning_tokens: usage
                    .pointer("/completion_tokens_details/reasoning_tokens")
                    .and_then(serde_json::Value::as_u64)
                    .unwrap_or_default(),
                cost: usage.get("cost").and_then(serde_json::Value::as_f64),
            },
        }));
    }
    for choice in value
        .get("choices")
        .and_then(serde_json::Value::as_array)
        .into_iter()
        .flatten()
    {
        let delta = choice.get("delta").unwrap_or(&serde_json::Value::Null);
        if let Some(text) = delta.get("content").and_then(serde_json::Value::as_str)
            && !text.is_empty()
        {
            events.push(Ok(StreamEvent::TextDelta { text: text.into() }));
        }
        if let Some(text) = delta
            .get("reasoning_content")
            .or_else(|| delta.get("reasoning"))
            .and_then(serde_json::Value::as_str)
            && !text.is_empty()
        {
            events.push(Ok(StreamEvent::ThinkingDelta { text: text.into() }));
        }
        for call in delta
            .get("tool_calls")
            .and_then(serde_json::Value::as_array)
            .into_iter()
            .flatten()
        {
            let index = call
                .get("index")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or_default() as usize;
            events.push(Ok(StreamEvent::ToolCallDelta {
                index,
                id: call
                    .get("id")
                    .and_then(serde_json::Value::as_str)
                    .map(str::to_string),
                name: call
                    .pointer("/function/name")
                    .and_then(serde_json::Value::as_str)
                    .map(str::to_string),
                arguments: call
                    .pointer("/function/arguments")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or_default()
                    .to_string(),
            }));
        }
        if let Some(reason) = choice
            .get("finish_reason")
            .and_then(serde_json::Value::as_str)
        {
            events.push(Ok(StreamEvent::Done {
                stop_reason: Some(reason.into()),
            }));
        }
    }
    events
}

fn parse_anthropic_event(event: &str, data: &str) -> Vec<Result<StreamEvent, LlmError>> {
    let value: serde_json::Value = match serde_json::from_str(data) {
        Ok(value) => value,
        Err(error) => return vec![Err(LlmError::Stream(error.to_string()))],
    };
    let parsed = match event {
        "content_block_delta" => match value
            .pointer("/delta/type")
            .and_then(serde_json::Value::as_str)
        {
            Some("text_delta") => value
                .pointer("/delta/text")
                .and_then(serde_json::Value::as_str)
                .map(|text| StreamEvent::TextDelta { text: text.into() }),
            Some("thinking_delta") => value
                .pointer("/delta/thinking")
                .and_then(serde_json::Value::as_str)
                .map(|text| StreamEvent::ThinkingDelta { text: text.into() }),
            Some("input_json_delta") => Some(StreamEvent::ToolCallDelta {
                index: value
                    .get("index")
                    .and_then(serde_json::Value::as_u64)
                    .unwrap_or_default() as usize,
                id: None,
                name: None,
                arguments: value
                    .pointer("/delta/partial_json")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or_default()
                    .into(),
            }),
            _ => None,
        },
        "content_block_start"
            if value
                .pointer("/content_block/type")
                .and_then(serde_json::Value::as_str)
                == Some("tool_use") =>
        {
            Some(StreamEvent::ToolCallDelta {
                index: value
                    .get("index")
                    .and_then(serde_json::Value::as_u64)
                    .unwrap_or_default() as usize,
                id: value
                    .pointer("/content_block/id")
                    .and_then(serde_json::Value::as_str)
                    .map(str::to_string),
                name: value
                    .pointer("/content_block/name")
                    .and_then(serde_json::Value::as_str)
                    .map(str::to_string),
                arguments: String::new(),
            })
        }
        "content_block_stop" => Some(StreamEvent::ToolCallComplete {
            index: value
                .get("index")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or_default() as usize,
        }),
        "message_start" => value
            .pointer("/message/usage")
            .map(|usage| StreamEvent::Usage {
                usage: anthropic_usage(usage),
            }),
        "message_delta" => {
            let usage = value.get("usage").map(anthropic_usage);
            let reason = value
                .pointer("/delta/stop_reason")
                .and_then(serde_json::Value::as_str)
                .map(str::to_string);
            if let Some(usage) = usage {
                return vec![
                    Ok(StreamEvent::Usage { usage }),
                    Ok(StreamEvent::Done {
                        stop_reason: reason,
                    }),
                ];
            }
            Some(StreamEvent::Done {
                stop_reason: reason,
            })
        }
        "message_stop" => Some(StreamEvent::Done { stop_reason: None }),
        "error" => {
            return vec![Err(LlmError::Stream(
                value
                    .pointer("/error/message")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or("Anthropic stream error")
                    .into(),
            ))];
        }
        _ => None,
    };
    parsed.into_iter().map(Ok).collect()
}

fn anthropic_usage(value: &serde_json::Value) -> Usage {
    Usage {
        input_tokens: value
            .get("input_tokens")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or_default(),
        output_tokens: value
            .get("output_tokens")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or_default(),
        cache_creation_input_tokens: value
            .get("cache_creation_input_tokens")
            .and_then(serde_json::Value::as_u64),
        cache_read_input_tokens: value
            .get("cache_read_input_tokens")
            .and_then(serde_json::Value::as_u64),
        reasoning_tokens: 0,
        cost: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_openai_text_tools_and_usage() {
        let events = parse_openai_event(
            r#"{"choices":[{"delta":{"content":"hi","tool_calls":[{"index":0,"id":"call_1","function":{"name":"search","arguments":"{}"}}]},"finish_reason":null}],"usage":{"prompt_tokens":3,"completion_tokens":1,"total_tokens":4}}"#,
        );
        assert!(matches!(events[0], Ok(StreamEvent::Usage { .. })));
        assert!(matches!(events[1], Ok(StreamEvent::TextDelta { .. })));
        assert!(matches!(events[2], Ok(StreamEvent::ToolCallDelta { .. })));
    }

    #[test]
    fn parses_anthropic_tool_input_delta() {
        let events = parse_anthropic_event(
            "content_block_delta",
            r#"{"index":1,"delta":{"type":"input_json_delta","partial_json":"{\"q\":\"rust\"}"}}"#,
        );
        assert!(matches!(
            events[0],
            Ok(StreamEvent::ToolCallDelta { index: 1, .. })
        ));
    }
}
