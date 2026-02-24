use serde_json::{Value, json};

use crate::types::openai::{InboundChatRequest, InboundContent, InboundContentPart};

/// Convert an inbound OpenAI chat request to an Anthropic request body (as JSON Value).
///
/// Handles:
/// - System message extraction (OpenAI role:"system" → Anthropic system field)
/// - Content format conversion (text/parts/null → Anthropic content blocks)
/// - tool_calls in assistant messages → tool_use content blocks
/// - role:"tool" messages → user messages with tool_result content blocks
/// - Tool definition format conversion
pub fn inbound_request_to_anthropic(req: InboundChatRequest) -> Value {
    let mut system_parts: Vec<Value> = Vec::new();
    let mut messages: Vec<Value> = Vec::new();

    for msg in req.messages {
        match msg.role.as_str() {
            "system" => {
                let text = match msg.content {
                    InboundContent::Text(t) => t,
                    InboundContent::Parts(parts) => parts
                        .into_iter()
                        .filter_map(|p| match p {
                            InboundContentPart::Text { text } => Some(text),
                            _ => None,
                        })
                        .collect::<Vec<_>>()
                        .join(""),
                    InboundContent::Null => String::new(),
                };
                if !text.is_empty() {
                    system_parts.push(json!({"type": "text", "text": text}));
                }
            }
            "tool" => {
                let content = match msg.content {
                    InboundContent::Text(t) => t,
                    _ => String::new(),
                };
                let tool_call_id = msg.tool_call_id.unwrap_or_default();
                messages.push(json!({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": tool_call_id,
                        "content": content
                    }]
                }));
            }
            "assistant" => {
                let mut content_blocks: Vec<Value> = Vec::new();

                // Add text content
                match msg.content {
                    InboundContent::Text(t) if !t.is_empty() => {
                        content_blocks.push(json!({"type": "text", "text": t}));
                    }
                    InboundContent::Parts(parts) => {
                        for part in parts {
                            match part {
                                InboundContentPart::Text { text } => {
                                    content_blocks.push(json!({"type": "text", "text": text}));
                                }
                                InboundContentPart::ImageUrl { image_url } => {
                                    // Pass through as image block
                                    if image_url.url.starts_with("data:") {
                                        // Base64 encoded image
                                        if let Some((media_type, data)) =
                                            parse_data_url(&image_url.url)
                                        {
                                            content_blocks.push(json!({
                                                "type": "image",
                                                "source": {
                                                    "type": "base64",
                                                    "media_type": media_type,
                                                    "data": data
                                                }
                                            }));
                                        }
                                    }
                                }
                            }
                        }
                    }
                    _ => {}
                }

                // Add tool_calls as tool_use blocks
                if let Some(tool_calls) = msg.tool_calls {
                    for tc in tool_calls {
                        let input: Value =
                            serde_json::from_str(&tc.function.arguments).unwrap_or(json!({}));
                        content_blocks.push(json!({
                            "type": "tool_use",
                            "id": tc.id,
                            "name": tc.function.name,
                            "input": input
                        }));
                    }
                }

                if !content_blocks.is_empty() {
                    messages.push(json!({
                        "role": "assistant",
                        "content": content_blocks
                    }));
                }
            }
            _ => {
                let content_blocks = match msg.content {
                    InboundContent::Text(t) => vec![json!({"type": "text", "text": t})],
                    InboundContent::Parts(parts) => parts
                        .into_iter()
                        .filter_map(|p| match p {
                            InboundContentPart::Text { text } => {
                                Some(json!({"type": "text", "text": text}))
                            }
                            InboundContentPart::ImageUrl { image_url } => {
                                if image_url.url.starts_with("data:") {
                                    parse_data_url(&image_url.url).map(|(media_type, data)| {
                                        json!({
                                            "type": "image",
                                            "source": {
                                                "type": "base64",
                                                "media_type": media_type,
                                                "data": data
                                            }
                                        })
                                    })
                                } else {
                                    None
                                }
                            }
                        })
                        .collect(),
                    InboundContent::Null => vec![json!({"type": "text", "text": ""})],
                };

                if !content_blocks.is_empty() {
                    messages.push(json!({
                        "role": "user",
                        "content": content_blocks
                    }));
                }
            }
        }
    }

    let mut body = json!({
        "messages": messages,
    });

    if let Some(model) = &req.model {
        body["model"] = json!(model);
    }
    if let Some(max_tokens) = req.max_tokens {
        body["max_tokens"] = json!(max_tokens);
    }
    if let Some(temperature) = req.temperature {
        body["temperature"] = json!(temperature);
    }
    if !system_parts.is_empty() {
        body["system"] = json!(system_parts);
    }

    // Convert tools
    if let Some(tools) = req.tools {
        let anthropic_tools: Vec<Value> = tools.into_iter().map(openai_tool_to_anthropic).collect();
        body["tools"] = json!(anthropic_tools);
    }

    body
}

/// Convert an OpenAI tool definition (JSON) to Anthropic format.
///
/// OpenAI: `{ "type": "function", "function": { "name", "description", "parameters" } }`
/// Anthropic: `{ "name", "description", "input_schema" }`
pub fn openai_tool_to_anthropic(tool: Value) -> Value {
    if let Some(function) = tool.get("function") {
        let name = function.get("name").cloned().unwrap_or(json!("unknown"));
        let description = function.get("description").cloned().unwrap_or(json!(""));
        let parameters = function
            .get("parameters")
            .cloned()
            .unwrap_or(json!({"type": "object", "properties": {}}));

        json!({
            "name": name,
            "description": description,
            "input_schema": parameters
        })
    } else {
        // Already in Anthropic format or unknown — pass through
        tool
    }
}

/// Parse a data URL into (media_type, base64_data).
fn parse_data_url(url: &str) -> Option<(String, String)> {
    let rest = url.strip_prefix("data:")?;
    let (header, data) = rest.split_once(',')?;
    let media_type = header.strip_suffix(";base64")?;
    Some((media_type.to_string(), data.to_string()))
}
