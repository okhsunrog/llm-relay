use serde_json::{Value, json};

/// Maximum number of cache_control blocks allowed by Anthropic API.
const MAX_CACHE_CONTROL_BLOCKS: usize = 4;

/// Inject cache_control breakpoints for optimal Anthropic prompt caching.
///
/// Per Anthropic docs, caching order is: tools -> system -> messages.
/// Up to 4 breakpoints allowed, each can reduce cost by 90% on cached tokens.
/// Respects the 4-block limit by counting existing blocks first.
pub fn ensure_cache_control(mut body: Value) -> Value {
    let existing = count_cache_control_blocks(&body);
    if existing >= MAX_CACHE_CONTROL_BLOCKS {
        return body;
    }

    let mut remaining = MAX_CACHE_CONTROL_BLOCKS - existing;

    // 1. Inject into last tool (caches all tool definitions)
    if remaining > 0 {
        let before = count_cache_control_blocks(&body);
        body = inject_tools_cache_control(body);
        let after = count_cache_control_blocks(&body);
        remaining -= after - before;
    }

    // 2. Inject into last system element
    if remaining > 0 {
        let before = count_cache_control_blocks(&body);
        body = inject_system_cache_control(body);
        let after = count_cache_control_blocks(&body);
        remaining -= after - before;
    }

    // 3. Inject into second-to-last user turn (multi-turn caching)
    if remaining > 0 {
        body = inject_messages_cache_control(body);
    }

    body
}

fn count_cache_control_blocks(body: &Value) -> usize {
    let mut count = 0;

    if let Some(Value::Array(arr)) = body.get("system") {
        for item in arr {
            if item.get("cache_control").is_some() {
                count += 1;
            }
        }
    }

    if let Some(Value::Array(tools)) = body.get("tools") {
        for tool in tools {
            if tool.get("cache_control").is_some() {
                count += 1;
            }
        }
    }

    if let Some(Value::Array(messages)) = body.get("messages") {
        for msg in messages {
            if let Some(Value::Array(content)) = msg.get("content") {
                for block in content {
                    if block.get("cache_control").is_some() {
                        count += 1;
                    }
                }
            }
        }
    }

    count
}

fn inject_tools_cache_control(mut body: Value) -> Value {
    let tools = match body.get_mut("tools").and_then(|t| t.as_array_mut()) {
        Some(arr) if !arr.is_empty() => arr,
        _ => return body,
    };

    if tools.iter().any(|t| t.get("cache_control").is_some()) {
        return body;
    }

    if let Some(last) = tools.last_mut()
        && let Some(obj) = last.as_object_mut()
    {
        obj.insert("cache_control".to_string(), json!({"type": "ephemeral"}));
    }

    body
}

fn inject_system_cache_control(mut body: Value) -> Value {
    let system = match body.get_mut("system") {
        Some(s) => s,
        None => return body,
    };

    match system {
        Value::Array(arr) if !arr.is_empty() => {
            if arr.iter().any(|s| s.get("cache_control").is_some()) {
                return body;
            }
            if let Some(last) = arr.last_mut()
                && let Some(obj) = last.as_object_mut()
            {
                obj.insert("cache_control".to_string(), json!({"type": "ephemeral"}));
            }
        }
        Value::String(text) => {
            let text = text.clone();
            *system = json!([{
                "type": "text",
                "text": text,
                "cache_control": {"type": "ephemeral"}
            }]);
        }
        _ => {}
    }

    body
}

fn inject_messages_cache_control(mut body: Value) -> Value {
    let messages = match body.get_mut("messages").and_then(|m| m.as_array_mut()) {
        Some(arr) => arr,
        None => return body,
    };

    let has_cache = messages.iter().any(|msg| {
        msg.get("content")
            .and_then(|c| c.as_array())
            .is_some_and(|arr| arr.iter().any(|b| b.get("cache_control").is_some()))
    });
    if has_cache {
        return body;
    }

    let user_indices: Vec<usize> = messages
        .iter()
        .enumerate()
        .filter(|(_, m)| m.get("role").and_then(|r| r.as_str()) == Some("user"))
        .map(|(i, _)| i)
        .collect();

    if user_indices.len() < 2 {
        return body;
    }

    let target_idx = user_indices[user_indices.len() - 2];

    if let Some(msg) = messages.get_mut(target_idx)
        && let Some(content) = msg.get_mut("content")
    {
        match content {
            Value::Array(arr) if !arr.is_empty() => {
                if let Some(last) = arr.last_mut()
                    && let Some(obj) = last.as_object_mut()
                {
                    obj.insert("cache_control".to_string(), json!({"type": "ephemeral"}));
                }
            }
            Value::String(text) => {
                let text = text.clone();
                *content = json!([{
                    "type": "text",
                    "text": text,
                    "cache_control": {"type": "ephemeral"}
                }]);
            }
            _ => {}
        }
    }

    body
}
