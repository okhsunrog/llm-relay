use serde_json::Value;

/// Add `mcp_` prefix to a tool name if not already present.
pub fn add_mcp_prefix(name: &str) -> String {
    if name.starts_with("mcp_") {
        name.to_string()
    } else {
        format!("mcp_{}", name)
    }
}

/// Strip `mcp_` prefix from a tool name if present.
pub fn strip_mcp_prefix(name: &str) -> String {
    name.strip_prefix("mcp_").unwrap_or(name).to_string()
}

/// Transform tool names in a request body for OAuth (add mcp_ prefix).
///
/// Transforms:
/// - Tool definitions in `tools` array (skipping built-in tools with `type` field)
/// - `tool_choice.name` when type is "tool"
/// - `tool_use` blocks in messages
pub fn transform_request_tool_names(body: &mut Value) {
    let obj = match body.as_object_mut() {
        Some(o) => o,
        None => return,
    };

    // Transform tools array (skip built-in tools with a "type" field)
    if let Some(Value::Array(tools)) = obj.get_mut("tools") {
        for tool in tools.iter_mut() {
            if tool
                .get("type")
                .and_then(|t| t.as_str())
                .is_some_and(|t| !t.is_empty())
            {
                continue;
            }

            if let Some(name) = tool
                .get("name")
                .and_then(|n| n.as_str())
                .map(|s| s.to_string())
                && let Some(obj) = tool.as_object_mut()
            {
                obj.insert("name".to_string(), Value::String(add_mcp_prefix(&name)));
            }
        }
    }

    // Transform tool_choice.name when type is "tool"
    if obj
        .get("tool_choice")
        .and_then(|tc| tc.get("type"))
        .and_then(|t| t.as_str())
        == Some("tool")
        && let Some(Value::Object(tool_choice)) = obj.get_mut("tool_choice")
        && let Some(name) = tool_choice
            .get("name")
            .and_then(|n| n.as_str())
            .map(|s| s.to_string())
        && !name.is_empty()
        && !name.starts_with("mcp_")
    {
        tool_choice.insert("name".to_string(), Value::String(add_mcp_prefix(&name)));
    }

    // Transform tool_use blocks in messages
    if let Some(Value::Array(messages)) = obj.get_mut("messages") {
        for msg in messages.iter_mut() {
            if let Some(Value::Array(content)) = msg.get_mut("content") {
                for block in content.iter_mut() {
                    if block.get("type").and_then(|t| t.as_str()) == Some("tool_use")
                        && let Some(name) = block
                            .get("name")
                            .and_then(|n| n.as_str())
                            .map(|s| s.to_string())
                        && let Some(obj) = block.as_object_mut()
                    {
                        obj.insert("name".to_string(), Value::String(add_mcp_prefix(&name)));
                    }
                }
            }
        }
    }
}

/// Transform tool names in a response body (strip mcp_ prefix).
///
/// Strips from `tool_use` blocks in the response content array.
pub fn transform_response_tool_names(body: &mut Value) {
    if let Some(Value::Array(content)) = body.get_mut("content") {
        for block in content.iter_mut() {
            if block.get("type").and_then(|t| t.as_str()) == Some("tool_use")
                && let Some(name) = block
                    .get("name")
                    .and_then(|n| n.as_str())
                    .map(|s| s.to_string())
                && let Some(obj) = block.as_object_mut()
            {
                obj.insert("name".to_string(), Value::String(strip_mcp_prefix(&name)));
            }
        }
    }
}
