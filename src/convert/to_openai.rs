use tracing::warn;

use crate::types::anthropic::{ContentBlock, Message, MessagesResponse};
use crate::types::common::{StopReason, ToolDefinition};
use crate::types::openai::{
    ChatMessage, ChatResponse, Choice, ResponseMessage, ResponseToolCall, ResponseToolCallFunction,
    ResponseUsage, Tool, ToolCallFunction, ToolCallOut, ToolFunction,
};

/// Convert Anthropic messages + system prompt to OpenAI message format.
///
/// - System prompt becomes the first `role: "system"` message
/// - Assistant text → `content`, ToolUse → `tool_calls` array
/// - User messages with ToolResult → multiple `role: "tool"` messages
/// - Thinking blocks are silently skipped
pub fn messages_to_openai(system: Option<&str>, messages: &[Message]) -> Vec<ChatMessage> {
    let mut out = Vec::new();

    if let Some(system) = system {
        out.push(ChatMessage::system(system));
    }

    for msg in messages {
        if msg.role == "assistant" {
            let text_parts: Vec<&str> = msg
                .content
                .iter()
                .filter_map(|b| {
                    if let ContentBlock::Text { text } = b {
                        Some(text.as_str())
                    } else {
                        None
                    }
                })
                .collect();

            let tool_calls: Vec<ToolCallOut> = msg
                .content
                .iter()
                .filter_map(|b| {
                    if let ContentBlock::ToolUse { id, name, input } = b {
                        Some(ToolCallOut {
                            id: id.clone(),
                            call_type: "function".to_string(),
                            function: ToolCallFunction {
                                name: name.clone(),
                                arguments: serde_json::to_string(input).unwrap_or_default(),
                            },
                        })
                    } else {
                        None
                    }
                })
                .collect();

            let content_str = if text_parts.is_empty() {
                None
            } else {
                Some(text_parts.join("\n"))
            };

            out.push(ChatMessage {
                role: "assistant".to_string(),
                content: content_str,
                tool_calls: if tool_calls.is_empty() {
                    None
                } else {
                    Some(tool_calls)
                },
                tool_call_id: None,
            });
        } else if msg.role == "user" {
            let has_tool_results = msg
                .content
                .iter()
                .any(|b| matches!(b, ContentBlock::ToolResult { .. }));

            if has_tool_results {
                for block in &msg.content {
                    if let ContentBlock::ToolResult {
                        tool_use_id,
                        content,
                        ..
                    } = block
                    {
                        out.push(ChatMessage::tool_result(tool_use_id, content));
                    }
                }
            } else {
                let text = msg
                    .content
                    .iter()
                    .filter_map(|b| {
                        if let ContentBlock::Text { text } = b {
                            Some(text.as_str())
                        } else {
                            None
                        }
                    })
                    .collect::<Vec<_>>()
                    .join("\n");

                out.push(ChatMessage::user(text));
            }
        }
    }

    out
}

/// Convert provider-agnostic ToolDefinitions to OpenAI tool format.
pub fn tools_to_openai(tools: &[ToolDefinition]) -> Vec<Tool> {
    tools
        .iter()
        .map(|t| Tool {
            tool_type: "function".to_string(),
            function: ToolFunction {
                name: t.name.clone(),
                description: t.description.clone(),
                parameters: t.input_schema.clone(),
            },
        })
        .collect()
}

/// Convert an OpenAI ChatResponse into an Anthropic MessagesResponse.
///
/// This is for projects that use Anthropic as canonical internal format —
/// they send to an OpenAI provider and need to normalize the response back.
pub fn response_to_anthropic(resp: ChatResponse) -> Result<MessagesResponse, String> {
    let choice = resp
        .choices
        .into_iter()
        .next()
        .ok_or("OpenAI response had no choices")?;

    let mut content: Vec<ContentBlock> = Vec::new();

    if let Some(text) = choice.message.content
        && !text.is_empty()
    {
        content.push(ContentBlock::Text { text });
    }

    if let Some(tool_calls) = choice.message.tool_calls {
        for tc in tool_calls {
            let input: serde_json::Value = serde_json::from_str(&tc.function.arguments)
                .unwrap_or_else(|e| {
                    warn!(
                        "Failed to parse tool arguments: {} — {}",
                        tc.function.arguments, e
                    );
                    serde_json::Value::Object(Default::default())
                });
            content.push(ContentBlock::ToolUse {
                id: tc.id,
                name: tc.function.name,
                input,
            });
        }
    }

    let stop_reason = StopReason::from_openai(choice.finish_reason.as_deref().unwrap_or("stop"));

    Ok(MessagesResponse {
        id: resp.id,
        model: resp.model,
        content,
        stop_reason,
        usage: resp.usage.map(|u| crate::types::common::Usage {
            input_tokens: u.prompt_tokens,
            output_tokens: u.completion_tokens,
            cache_creation_input_tokens: u.cache_creation_input_tokens,
            cache_read_input_tokens: u.cache_read_input_tokens,
        }),
    })
}

/// Convert an Anthropic MessagesResponse to an OpenAI ChatResponse.
///
/// This is for proxy scenarios — Anthropic response → OpenAI format out.
pub fn anthropic_response_to_openai(resp: MessagesResponse) -> ChatResponse {
    let mut text_parts = Vec::new();
    let mut reasoning_parts = Vec::new();
    let mut tool_calls = Vec::new();

    for block in &resp.content {
        match block {
            ContentBlock::Text { text } => text_parts.push(text.as_str()),
            ContentBlock::Thinking { thinking, .. } => reasoning_parts.push(thinking.as_str()),
            ContentBlock::ToolUse { id, name, input } => {
                tool_calls.push(ResponseToolCall {
                    id: id.clone(),
                    call_type: Some("function".to_string()),
                    function: ResponseToolCallFunction {
                        name: name.clone(),
                        arguments: serde_json::to_string(input).unwrap_or_default(),
                    },
                });
            }
            _ => {}
        }
    }

    let content = if text_parts.is_empty() {
        None
    } else {
        Some(text_parts.join(""))
    };

    let reasoning_content = if reasoning_parts.is_empty() {
        None
    } else {
        Some(reasoning_parts.join(""))
    };

    let finish_reason = resp.stop_reason.to_openai();

    let usage = resp.usage.map(|u| ResponseUsage {
        prompt_tokens: u.input_tokens,
        completion_tokens: u.output_tokens,
        total_tokens: u.input_tokens + u.output_tokens,
        cache_creation_input_tokens: u.cache_creation_input_tokens,
        cache_read_input_tokens: u.cache_read_input_tokens,
    });

    ChatResponse {
        id: resp.id.clone(),
        object: Some("chat.completion".to_string()),
        created: Some(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
        ),
        model: resp.model.clone(),
        choices: vec![Choice {
            index: Some(0),
            message: ResponseMessage {
                role: Some("assistant".to_string()),
                content,
                reasoning_content,
                tool_calls: if tool_calls.is_empty() {
                    None
                } else {
                    Some(tool_calls)
                },
            },
            finish_reason: Some(finish_reason.to_string()),
        }],
        usage,
    }
}
