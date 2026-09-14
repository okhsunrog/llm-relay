use super::*;
use serde_json::{Value, json};

#[test]
fn native_requests_retain_unknown_fields_and_blocks() {
    for protocol in [
        Protocol::Messages,
        Protocol::ChatCompletions,
        Protocol::Responses,
    ] {
        let body = json!({"model":"x","future_feature":{"unknown":true},"input":[{"type":"future_item","blob":"opaque"}]});
        assert_eq!(
            translate_request(protocol, protocol, &body).unwrap().body,
            body
        );
    }
}
#[test]
fn request_matrix_preserves_text_tools_and_multimodal_results() {
    let source = json!({"model":"gpt-test","system":"Be concise","max_tokens":100,"messages":[
        {"role":"user","content":[{"type":"text","text":"Look"},{"type":"image","source":{"type":"base64","media_type":"image/png","data":"abc"}}]},
        {"role":"assistant","content":[{"type":"tool_use","id":"call_a","name":"read","input":{"x":1}},{"type":"tool_use","id":"call_b","name":"read","input":{"x":2}}]},
        {"role":"user","content":[{"type":"tool_result","tool_use_id":"call_a","content":[{"type":"text","text":"one"},{"type":"image","source":{"type":"url","url":"https://example.com/a.png"}}]},{"type":"tool_result","tool_use_id":"call_b","content":"two"}]}],
        "tools":[{"name":"read","description":"Read value","input_schema":{"type":"object","properties":{"x":{"type":"integer"}}}}]});
    let canonical = decode_request(Protocol::Messages, &source).unwrap();
    for target in [
        Protocol::Messages,
        Protocol::ChatCompletions,
        Protocol::Responses,
    ] {
        let encoded = encode_request(&canonical, target).unwrap();
        let roundtrip = decode_request(target, &encoded.body).unwrap();
        assert_eq!(roundtrip.items, canonical.items, "{target:?}");
        assert_eq!(roundtrip.tools, canonical.tools);
    }
}
#[test]
fn cross_protocol_unknown_features_are_rejected() {
    for extra in [
        json!({"unsupported":true}),
        json!({"messages":[{"role":"user","content":[{"type":"future_block"}]}]}),
        json!({"tools":[{"type":"web_search_20250305","name":"search"}]}),
    ] {
        let mut body = json!({"model":"gpt-test","messages":[{"role":"user","content":"hi"}]});
        body.as_object_mut()
            .unwrap()
            .extend(extra.as_object().unwrap().clone());
        assert!(translate_request(Protocol::Messages, Protocol::Responses, &body).is_err());
    }
}
#[test]
fn thinking_state_retains_origin_and_round_trips() {
    let reason = json!({"type":"reasoning","id":"rs_1","summary":[{"type":"summary_text","text":"Summary"}],"encrypted_content":"gpt-opaque"});
    let source =
        json!({"model":"gpt-test","input":[reason.clone(),{"role":"user","content":"Continue"}]});
    let messages = translate_request(Protocol::Responses, Protocol::Messages, &source).unwrap();
    let back = translate_request(Protocol::Messages, Protocol::Responses, &messages.body).unwrap();
    assert_eq!(back.body["input"][0], reason);
    let foreign = json!({"model":"gpt-test","messages":[{"role":"assistant","content":[{"type":"thinking","thinking":"summary","signature":"claude-signature"}]}]});
    assert!(
        translate_request(Protocol::Messages, Protocol::Responses, &foreign)
            .unwrap_err()
            .contains("Opaque reasoning")
    );
}
#[test]
fn loss_policy_is_explicit() {
    let source = json!({"model":"gpt-test","thinking":{"type":"enabled","budget_tokens":4096},"messages":[{"role":"user","content":[{"type":"text","text":"Hi","cache_control":{"type":"ephemeral"}}]}]});
    let converted = translate_request(Protocol::Messages, Protocol::Responses, &source).unwrap();
    assert_eq!(converted.body["reasoning"]["effort"], "medium");
    assert!(converted.clone().enforce(Policy::Strict).is_err());
    assert!(converted.enforce(Policy::Compatible).is_ok());
}
fn decoded(d: &mut Decoder, event: Value) -> events::Decoded {
    d.decode(event).unwrap()
}
#[test]
fn interleaved_tools_produce_dense_blocks_and_valid_json_completion() {
    let mut d = Decoder::default();
    let mut e = Encoder::new(Protocol::Messages, "gpt-test".into(), false);
    let mut frames = vec![];
    let events = vec![
        json!({"type":"response.created","response":{"id":"r","model":"gpt-test","created_at":1}}),
        json!({"type":"response.output_item.added","output_index":2,"item":{"type":"function_call","call_id":"a","name":"read","arguments":""}}),
        json!({"type":"response.output_item.added","output_index":5,"item":{"type":"function_call","call_id":"b","name":"read","arguments":""}}),
        json!({"type":"response.function_call_arguments.delta","output_index":5,"delta":"{\"x\":"}),
        json!({"type":"response.function_call_arguments.delta","output_index":2,"delta":"{\"x\":1}"}),
        json!({"type":"response.function_call_arguments.delta","output_index":5,"delta":"2}"}),
        json!({"type":"response.output_item.done","output_index":5,"item":{"type":"function_call","call_id":"b","name":"read","arguments":"{\"x\":2}"}}),
        json!({"type":"response.output_item.done","output_index":2,"item":{"type":"function_call","call_id":"a","name":"read","arguments":"{\"x\":1}"}}),
        json!({"type":"response.completed","response":{"id":"r","model":"gpt-test","status":"completed","output":[],"usage":{"input_tokens":10,"output_tokens":5,"total_tokens":15}}}),
    ];
    let mut final_message = Value::Null;
    for raw in events {
        let event = decoded(&mut d, raw);
        if let Event::Finish(c) = &event.event {
            final_message = c.render(Protocol::Messages).unwrap();
        }
        frames.extend(e.encode(&event).unwrap());
    }
    let starts = frames
        .iter()
        .filter(|f| f.data["type"] == "content_block_start")
        .collect::<Vec<_>>();
    assert_eq!(starts.len(), 2);
    assert_eq!(starts[0].data["index"], 0);
    assert_eq!(starts[1].data["index"], 1);
    for (index, expected) in [(0, json!({"x":1})), (1, json!({"x":2}))] {
        let arguments = frames
            .iter()
            .filter(|f| f.data["type"] == "content_block_delta" && f.data["index"] == index)
            .filter_map(|f| f.data["delta"]["partial_json"].as_str())
            .collect::<String>();
        assert_eq!(serde_json::from_str::<Value>(&arguments).unwrap(), expected);
    }
    assert_eq!(final_message["content"][0]["id"], "a");
    assert_eq!(final_message["content"][1]["id"], "b");
    assert_eq!(final_message["stop_reason"], "tool_use");
    assert_eq!(frames.last().unwrap().data["type"], "message_stop");
    assert!(d.finish().is_ok());
}
#[test]
fn truncated_or_failed_streams_are_not_successes() {
    let mut d = Decoder::default();
    assert!(d.finish().is_err());
    assert!(d.decode(json!({"type":"response.failed","response":{"usage":{"input_tokens":5,"output_tokens":2}}})).is_err());
    assert_eq!(d.usage().unwrap().input_tokens, 5);
}
#[test]
fn no_features_build_still_has_protocol_codec() {
    let response = json!({"id":"r","model":"gpt","status":"completed","output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"Привет"}]}],"usage":{"input_tokens":10,"input_tokens_details":{"cached_tokens":5,"cache_write_tokens":2},"output_tokens":4,"total_tokens":14}});
    let message = translate_response(Protocol::Responses, Protocol::Messages, &response).unwrap();
    assert_eq!(message["content"][0]["text"], "Привет");
    assert_eq!(message["usage"]["input_tokens"], 3);
    let chat =
        translate_response(Protocol::Responses, Protocol::ChatCompletions, &response).unwrap();
    assert_eq!(chat["usage"]["prompt_tokens"], 10);
}

#[test]
fn native_unknown_output_survives_but_cross_format_rejects() {
    let body = json!({"id":"r","status":"completed","output":[{"type":"message","role":"assistant","content":[{"type":"future_audio","data":"opaque"}]}]});
    let mut decoder = Decoder::default();
    let decoded = decoder
        .decode(json!({"type":"response.completed","response":body}))
        .unwrap();
    let Event::Finish(done) = decoded.event else {
        panic!("completion expected")
    };
    assert_eq!(done.render(Protocol::Responses).unwrap(), body);
    assert!(done.render(Protocol::ChatCompletions).is_err());
    assert!(done.render(Protocol::Messages).is_err());
}

#[test]
fn system_cache_hint_is_reported() {
    let body = json!({"model":"m","system":[{"type":"text","text":"system","cache_control":{"type":"ephemeral"}}],"messages":[{"role":"user","content":"hi"}]});
    let translated = translate_request(Protocol::Messages, Protocol::Responses, &body).unwrap();
    assert!(translated.enforce(Policy::Strict).is_err());
}

#[test]
fn terminal_only_unknown_items_do_not_produce_stream_success() {
    let mut decoder = Decoder::default();
    let decoded = decoder
        .decode(json!({"type":"response.completed","response":{"output":[{"type":"future_item"}]}}))
        .unwrap();
    for protocol in [Protocol::Messages, Protocol::ChatCompletions] {
        assert!(
            Encoder::new(protocol, "m".into(), false)
                .encode(&decoded)
                .is_err()
        );
    }
    assert!(
        Encoder::new(Protocol::Responses, "m".into(), false)
            .encode(&decoded)
            .is_ok()
    );
}

#[test]
fn filtered_completion_has_same_json_and_stream_stop_reason() {
    let mut decoder = Decoder::default();
    let decoded = decoder.decode(json!({"type":"response.incomplete","response":{"status":"incomplete","incomplete_details":{"reason":"content_filter"},"output":[]}})).unwrap();
    let Event::Finish(c) = &decoded.event else {
        panic!("completion expected")
    };
    let frames = Encoder::new(Protocol::ChatCompletions, "m".into(), false)
        .encode(&decoded)
        .unwrap();
    assert_eq!(
        frames[0].data["choices"][0]["finish_reason"],
        "content_filter"
    );
    assert_eq!(
        c.render(Protocol::ChatCompletions).unwrap()["choices"][0]["finish_reason"],
        "content_filter"
    );
}
