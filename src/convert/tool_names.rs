use serde_json::Value;
use std::collections::HashSet;

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
///
/// Built-in/server tools (those carrying a `type` field, e.g. `web_search`,
/// `advisor`) are left unprefixed everywhere. Prefixing only their `tool_use`
/// or `tool_choice` references — while leaving their definitions unprefixed —
/// would make the name not match the tool (`mcp_web_search` vs `web_search`).
pub fn transform_request_tool_names(body: &mut Value) {
    let obj = match body.as_object_mut() {
        Some(o) => o,
        None => return,
    };

    // Names of built-in/typed tools, whose definitions stay unprefixed and so
    // must not be prefixed in tool_choice or message history either.
    let mut typed: HashSet<String> = HashSet::new();

    // Transform tools array (skip built-in tools with a "type" field)
    if let Some(Value::Array(tools)) = obj.get_mut("tools") {
        for tool in tools.iter_mut() {
            if tool
                .get("type")
                .and_then(|t| t.as_str())
                .is_some_and(|t| !t.is_empty())
            {
                if let Some(name) = tool.get("name").and_then(|n| n.as_str()) {
                    typed.insert(name.to_string());
                }
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

    // Transform tool_choice.name when type is "tool" (skip built-in/typed tools)
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
        && !typed.contains(&name)
    {
        tool_choice.insert("name".to_string(), Value::String(add_mcp_prefix(&name)));
    }

    // Transform tool_use blocks in messages (skip built-in/typed tools)
    if let Some(Value::Array(messages)) = obj.get_mut("messages") {
        for msg in messages.iter_mut() {
            if let Some(Value::Array(content)) = msg.get_mut("content") {
                for block in content.iter_mut() {
                    if block.get("type").and_then(|t| t.as_str()) == Some("tool_use")
                        && let Some(name) = block
                            .get("name")
                            .and_then(|n| n.as_str())
                            .map(|s| s.to_string())
                        && !typed.contains(&name)
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

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn prefixes_regular_tools_everywhere() {
        let mut body = json!({
            "tools": [{"name": "Read"}],
            "tool_choice": {"type": "tool", "name": "Read"},
            "messages": [{
                "role": "assistant",
                "content": [{"type": "tool_use", "id": "a", "name": "Read", "input": {}}]
            }]
        });

        transform_request_tool_names(&mut body);

        assert_eq!(body["tools"][0]["name"], "mcp_Read");
        assert_eq!(body["tool_choice"]["name"], "mcp_Read");
        assert_eq!(body["messages"][0]["content"][0]["name"], "mcp_Read");
    }

    #[test]
    fn leaves_typed_tools_unprefixed_in_history_and_choice() {
        // Built-in/server tools carry a `type` field; their definitions are not
        // prefixed, so their tool_use/tool_choice references must not be either
        // (regression: history was prefixed to `mcp_web_search`, which then
        // didn't match the `web_search` definition -> Anthropic 400).
        let mut body = json!({
            "tools": [
                {"type": "web_search_20250305", "name": "web_search"},
                {"name": "Read"}
            ],
            "tool_choice": {"type": "tool", "name": "web_search"},
            "messages": [{
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "a", "name": "web_search", "input": {}},
                    {"type": "tool_use", "id": "b", "name": "Read", "input": {}}
                ]
            }]
        });

        transform_request_tool_names(&mut body);

        // Typed tool stays bare in definition, tool_choice and history.
        assert_eq!(body["tools"][0]["name"], "web_search");
        assert_eq!(body["tool_choice"]["name"], "web_search");
        assert_eq!(body["messages"][0]["content"][0]["name"], "web_search");
        // A normal tool is still prefixed.
        assert_eq!(body["tools"][1]["name"], "mcp_Read");
        assert_eq!(body["messages"][0]["content"][1]["name"], "mcp_Read");
    }
}
