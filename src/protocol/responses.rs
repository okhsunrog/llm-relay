//! Stateless conversion of completed Responses output and usage.
// JSON reads return Null for missing fields; writes below target validated or constructed objects.
#![allow(clippy::indexing_slicing)]
use crate::Usage;
use serde_json::{Value, json};
use std::collections::BTreeMap;

pub fn usage(response: &Value) -> Option<Usage> {
    let u = response.get("usage").filter(|v| v.is_object())?;
    let input = u.get("input_tokens").and_then(Value::as_u64)?;
    let cached = u
        .pointer("/input_tokens_details/cached_tokens")
        .and_then(Value::as_u64)
        .unwrap_or(0)
        .min(input);
    let written = u
        .pointer("/input_tokens_details/cache_write_tokens")
        .and_then(Value::as_u64)
        .unwrap_or(0)
        .min(input.saturating_sub(cached));
    Some(Usage {
        input_tokens: input.saturating_sub(cached).saturating_sub(written),
        output_tokens: u.get("output_tokens").and_then(Value::as_u64).unwrap_or(0),
        cache_read_input_tokens: Some(cached),
        cache_creation_input_tokens: Some(written),
        reasoning_tokens: u
            .pointer("/output_tokens_details/reasoning_tokens")
            .and_then(Value::as_u64)
            .unwrap_or(0),
        ..Usage::default()
    })
}
fn chat_usage(response: &Value) -> Value {
    let u = &response["usage"];
    json!({"prompt_tokens":u["input_tokens"],"completion_tokens":u["output_tokens"],"total_tokens":u["total_tokens"],"prompt_tokens_details":u["input_tokens_details"],"completion_tokens_details":u["output_tokens_details"]})
}
fn finish_reason(response: &Value, tools: bool) -> &'static str {
    if response["incomplete_details"]["reason"] == "content_filter" {
        "content_filter"
    } else if response["status"] == "incomplete" {
        "length"
    } else if tools {
        "tool_calls"
    } else {
        "stop"
    }
}
pub fn to_chat(response: &Value) -> Value {
    let mut text = String::new();
    let mut refusal = String::new();
    let mut calls = Vec::new();
    if let Some(output) = response["output"].as_array() {
        for item in output {
            if item["type"] == "function_call" {
                calls.push(json!({"id":item["call_id"],"type":"function","function":{"name":item["name"],"arguments":item["arguments"]}}));
            }
            if let Some(parts) = item["content"].as_array() {
                for part in parts {
                    if part["type"] == "output_text" {
                        text.push_str(part["text"].as_str().unwrap_or(""));
                    }
                    if part["type"] == "refusal" {
                        refusal.push_str(part["refusal"].as_str().unwrap_or(""));
                    }
                }
            }
        }
    }
    let mut message =
        json!({"role":"assistant","content":if text.is_empty(){Value::Null}else{json!(text)}});
    if !calls.is_empty() {
        message["tool_calls"] = json!(calls);
    }
    if !refusal.is_empty() {
        message["refusal"] = json!(refusal);
    }
    json!({"id":response["id"],"object":"chat.completion","created":response["created_at"],"model":response["model"],"choices":[{"index":0,"message":message,"finish_reason":finish_reason(response,!calls.is_empty())}],"usage":chat_usage(response)})
}

/// Terminal responses sometimes omit output; retain completed items by output index.
#[derive(Default)]
pub struct Accumulator {
    items: BTreeMap<u64, Value>,
}
impl Accumulator {
    pub fn observe(&mut self, event: &mut Value) {
        if event["type"] == "response.output_item.done"
            && let Some(index) = event["output_index"].as_u64()
        {
            self.items.insert(index, event["item"].clone());
        }
        if matches!(
            event["type"].as_str(),
            Some("response.completed" | "response.incomplete" | "response.failed")
        ) && event
            .pointer("/response/output")
            .and_then(Value::as_array)
            .is_none_or(Vec::is_empty)
            && let Some(response) = event.get_mut("response").and_then(Value::as_object_mut)
        {
            response.insert(
                "output".into(),
                json!(self.items.values().collect::<Vec<_>>()),
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn completes_missing_output_in_order_and_preserves_native_items() {
        let mut a = Accumulator::default();
        a.observe(&mut json!({"type":"response.output_item.done","output_index":1,"item":{"type":"message","content":[{"type":"output_text","text":"Привет"}]}}));
        a.observe(&mut json!({"type":"response.output_item.done","output_index":0,"item":{"type":"reasoning","encrypted_content":"opaque"}}));
        let mut e = json!({"type":"response.completed","response":{"output":[]}});
        a.observe(&mut e);
        assert_eq!(
            e.pointer("/response/output/0/encrypted_content"),
            Some(&json!("opaque"))
        );
        assert_eq!(
            to_chat(&e["response"]).pointer("/choices/0/message/content"),
            Some(&json!("Привет"))
        );
    }
    #[test]
    fn cached_tokens_are_not_double_counted() {
        let usage=usage(&json!({"usage":{"input_tokens":100,"input_tokens_details":{"cached_tokens":80,"cache_write_tokens":10},"output_tokens":12}})).unwrap();
        assert_eq!(usage.input_tokens, 10);
        assert_eq!(usage.cache_creation_input_tokens, Some(10));
        assert_eq!(usage.cache_read_input_tokens, Some(80));
        assert_eq!(usage.output_tokens, 12);
    }
}
