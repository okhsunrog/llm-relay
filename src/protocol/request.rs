use super::model::*;
use base64::{Engine, engine::general_purpose::URL_SAFE_NO_PAD};
use serde_json::{Map, Value, json};

const SIGNATURE_PREFIX: &str = "llm-relay:responses:v1:";
pub fn reasoning_signature(value: &Value) -> String {
    format!(
        "{SIGNATURE_PREFIX}{}",
        URL_SAFE_NO_PAD.encode(value.to_string())
    )
}
fn text(v: &Value, key: &str) -> Result<String, String> {
    v.get(key)
        .and_then(Value::as_str)
        .map(str::to_owned)
        .ok_or_else(|| format!("Missing or invalid {key}"))
}
fn diagnose(r: &mut Request, field: &str, reason: &str) {
    r.diagnostics.push(Diagnostic {
        field: field.into(),
        reason: reason.into(),
    });
}
fn text_content(text: String) -> Vec<Content> {
    vec![Content::Text { text }]
}
fn parts(value: &Value, protocol: Protocol) -> Result<Vec<Content>, String> {
    if let Some(s) = value.as_str() {
        return Ok(text_content(s.into()));
    }
    if value.is_null() {
        return Ok(vec![]);
    }
    value
        .as_array()
        .ok_or("Content must be text or an array")?
        .iter()
        .map(|part| match part["type"].as_str().unwrap_or("") {
            "text" | "input_text" | "output_text" => Ok(Content::Text {
                text: text(part, "text")?,
            }),
            "refusal" => Ok(Content::Refusal {
                text: text(part, "refusal")?,
            }),
            "image_url" => Ok(Content::Image {
                url: text(&part["image_url"], "url")?,
                detail: part["image_url"]["detail"]
                    .as_str()
                    .filter(|v| *v != "auto")
                    .map(str::to_owned),
            }),
            "input_image" => Ok(Content::Image {
                url: text(part, "image_url")?,
                detail: part["detail"]
                    .as_str()
                    .filter(|v| *v != "auto")
                    .map(str::to_owned),
            }),
            "image" if protocol == Protocol::Messages => {
                let source = &part["source"];
                let url = match source["type"].as_str() {
                    Some("url") => text(source, "url")?,
                    Some("base64") => format!(
                        "data:{};base64,{}",
                        text(source, "media_type")?,
                        text(source, "data")?
                    ),
                    _ => return Err("Unsupported image source".into()),
                };
                Ok(Content::Image { url, detail: None })
            }
            "document" if protocol == Protocol::Messages => {
                let source = &part["source"];
                match source["type"].as_str() {
                    Some("text") => Ok(Content::Text {
                        text: text(source, "data")?,
                    }),
                    Some("base64") => Ok(Content::File {
                        data: format!(
                            "data:{};base64,{}",
                            text(source, "media_type")?,
                            text(source, "data")?
                        ),
                        filename: Some(part["title"].as_str().unwrap_or("document.pdf").into()),
                    }),
                    _ => Err(
                        "Only inline text or base64 documents are supported across protocols"
                            .into(),
                    ),
                }
            }
            "input_file" => Ok(Content::File {
                data: text(part, "file_data")?,
                filename: part["filename"].as_str().map(str::to_owned),
            }),
            _ => Err(format!("Unsupported content type: {}", part["type"])),
        })
        .collect()
}
fn message(r: &mut Request, role: &str, content: Vec<Content>) {
    if !content.is_empty() {
        r.items.push(Item::Message {
            role: role.into(),
            content,
        });
    }
}
fn copy_controls(r: &mut Request, body: &Value, fields: &[(&str, &str)]) {
    for (source, target) in fields {
        if let Some(v) = body.get(*source).filter(|v| !v.is_null()) {
            r.controls.insert((*target).into(), v.clone());
        }
    }
}
fn check_fields(body: &Value, allowed: &[&str]) -> Result<(), String> {
    for (field, value) in body.as_object().ok_or("Request must be an object")? {
        if !allowed.contains(&field.as_str()) && !value.is_null() {
            return Err(format!("Unsupported cross-protocol request field: {field}"));
        }
    }
    Ok(())
}
pub fn decode_request(protocol: Protocol, body: &Value) -> Result<Request, String> {
    let mut r = Request {
        model: text(body, "model")?,
        items: vec![],
        tools: vec![],
        controls: Map::new(),
        diagnostics: vec![],
    };
    copy_controls(
        &mut r,
        body,
        &[
            ("stream", "stream"),
            ("store", "store"),
            ("temperature", "temperature"),
            ("top_p", "top_p"),
            ("prompt_cache_key", "prompt_cache_key"),
            ("service_tier", "service_tier"),
        ],
    );
    match protocol {
        Protocol::Messages => {
            check_fields(
                body,
                &[
                    "model",
                    "messages",
                    "system",
                    "tools",
                    "tool_choice",
                    "max_tokens",
                    "stream",
                    "temperature",
                    "top_p",
                    "top_k",
                    "stop_sequences",
                    "thinking",
                    "output_config",
                    "metadata",
                    "cache_control",
                    "service_tier",
                ],
            )?;
            if let Some(system) = body.get("system") {
                message(&mut r, "developer", parts(system, protocol)?);
                if system
                    .as_array()
                    .is_some_and(|a| a.iter().any(|p| p.get("cache_control").is_some()))
                {
                    diagnose(
                        &mut r,
                        "system.cache_control",
                        "Cache hints unavailable in target protocol",
                    );
                }
            }
            for m in body["messages"]
                .as_array()
                .ok_or("messages must be an array")?
            {
                let role = text(m, "role")?;
                if !["user", "assistant"].contains(&role.as_str()) {
                    return Err("Messages role must be user or assistant".into());
                }
                if let Some(s) = m["content"].as_str() {
                    message(&mut r, &role, text_content(s.into()));
                    continue;
                }
                let mut pending = Vec::new();
                for part in m["content"]
                    .as_array()
                    .ok_or("Message content must be text or blocks")?
                {
                    match part["type"].as_str().unwrap_or("") {
                        "tool_use" => {
                            if role != "assistant" {
                                return Err("tool_use requires assistant role".into());
                            }
                            message(&mut r, &role, std::mem::take(&mut pending));
                            let args = part
                                .get("input")
                                .filter(|v| v.is_object())
                                .ok_or("Tool input must be an object")?;
                            r.items.push(Item::ToolCall {
                                id: text(part, "id")?,
                                name: text(part, "name")?,
                                arguments: args.to_string(),
                            });
                        }
                        "tool_result" => {
                            if role != "user" {
                                return Err("tool_result requires user role".into());
                            }
                            message(&mut r, &role, std::mem::take(&mut pending));
                            r.items.push(Item::ToolResult {
                                id: text(part, "tool_use_id")?,
                                content: parts(&part["content"], protocol)?,
                                is_error: part["is_error"].as_bool().unwrap_or(false),
                            });
                        }
                        "thinking" => {
                            if role != "assistant" {
                                return Err("thinking requires assistant role".into());
                            }
                            message(&mut r, &role, std::mem::take(&mut pending));
                            let opaque = match part["signature"].as_str() {
                                Some(s) if s.starts_with(SIGNATURE_PREFIX) => {
                                    let bytes = URL_SAFE_NO_PAD
                                        .decode(s.trim_start_matches(SIGNATURE_PREFIX))
                                        .map_err(|_| "Malformed reasoning envelope")?;
                                    let value: Value = serde_json::from_slice(&bytes)
                                        .map_err(|_| "Malformed reasoning envelope")?;
                                    if value["type"] != "reasoning"
                                        || !value["encrypted_content"].is_string()
                                    {
                                        return Err("Invalid reasoning envelope".into());
                                    }
                                    Some(Opaque {
                                        protocol: Protocol::Responses,
                                        value,
                                    })
                                }
                                Some(_) => Some(Opaque {
                                    protocol: Protocol::Messages,
                                    value: part.clone(),
                                }),
                                None => None,
                            };
                            r.items.push(Item::Reasoning {
                                summary: part["thinking"].as_str().unwrap_or("").into(),
                                opaque,
                            });
                        }
                        _ => pending.extend(parts(&json!([part]), protocol)?),
                    }
                    if part.get("cache_control").is_some() {
                        diagnose(
                            &mut r,
                            "cache_control",
                            "Source cache breakpoints do not transfer",
                        );
                    }
                }
                message(&mut r, &role, pending);
            }
            copy_controls(
                &mut r,
                body,
                &[
                    ("max_tokens", "max_output_tokens"),
                    ("stop_sequences", "stop"),
                    ("top_k", "top_k"),
                ],
            );
            if let Some(t) = body.get("thinking") {
                let effort = match t["type"].as_str() {
                    Some("disabled") => "none".to_owned(),
                    Some("adaptive" | "auto") => body
                        .pointer("/output_config/effort")
                        .and_then(Value::as_str)
                        .unwrap_or("high")
                        .into(),
                    Some("enabled") => {
                        let n = t["budget_tokens"]
                            .as_u64()
                            .ok_or("thinking budget must be an integer")?;
                        diagnose(
                            &mut r,
                            "thinking.budget_tokens",
                            "Token budget approximated by reasoning effort",
                        );
                        if n <= 1024 {
                            "low"
                        } else if n <= 8192 {
                            "medium"
                        } else {
                            "high"
                        }
                        .into()
                    }
                    _ => return Err("Unsupported thinking configuration".into()),
                };
                r.controls
                    .insert("reasoning".into(), json!({"effort":effort}));
            } else if let Some(effort) = body.pointer("/output_config/effort") {
                r.controls
                    .insert("reasoning".into(), json!({"effort":effort}));
            }
            if let Some(format) = body.pointer("/output_config/format") {
                let mut f = format.clone();
                if !f.is_object() {
                    return Err("Invalid output format".into());
                }
                f["name"] = json!("response");
                r.controls.insert("text".into(), json!({"format":f}));
            }
            if let Some(choice) = body.get("tool_choice") {
                let c = match choice["type"].as_str() {
                    Some("auto") => json!("auto"),
                    Some("any") => json!("required"),
                    Some("none") => json!("none"),
                    Some("tool") => json!({"type":"function","name":text(choice,"name")?}),
                    _ => return Err("Unsupported tool_choice".into()),
                };
                r.controls.insert("tool_choice".into(), c);
                if let Some(disable) = choice["disable_parallel_tool_use"].as_bool() {
                    r.controls
                        .insert("parallel_tool_calls".into(), json!(!disable));
                }
            }
            for field in ["metadata", "cache_control"] {
                if body.get(field).is_some() {
                    diagnose(&mut r, field, "Source hint does not transfer");
                }
            }
        }
        Protocol::ChatCompletions => {
            check_fields(
                body,
                &[
                    "model",
                    "messages",
                    "tools",
                    "tool_choice",
                    "stream",
                    "stream_options",
                    "store",
                    "max_tokens",
                    "max_completion_tokens",
                    "temperature",
                    "top_p",
                    "stop",
                    "reasoning_effort",
                    "response_format",
                    "parallel_tool_calls",
                    "n",
                    "user",
                    "prompt_cache_key",
                    "service_tier",
                ],
            )?;
            if body
                .get("n")
                .is_some_and(|v| !v.is_null() && v.as_u64() != Some(1))
            {
                return Err("Only n=1 is supported".into());
            }
            for m in body["messages"]
                .as_array()
                .ok_or("messages must be an array")?
            {
                let role = text(m, "role")?;
                if role == "tool" {
                    r.items.push(Item::ToolResult {
                        id: text(m, "tool_call_id")?,
                        content: parts(&m["content"], protocol)?,
                        is_error: false,
                    });
                    continue;
                }
                if !["system", "developer", "user", "assistant"].contains(&role.as_str()) {
                    return Err("Unsupported message role".into());
                }
                message(
                    &mut r,
                    if role == "system" { "developer" } else { &role },
                    parts(&m["content"], protocol)?,
                );
                if let Some(calls) = m.get("tool_calls").filter(|v| !v.is_null()) {
                    if role != "assistant" {
                        return Err("tool_calls requires assistant role".into());
                    }
                    for call in calls.as_array().ok_or("tool_calls must be an array")? {
                        if call["type"] != "function" {
                            return Err("Only function calls supported".into());
                        }
                        r.items.push(Item::ToolCall {
                            id: text(call, "id")?,
                            name: text(&call["function"], "name")?,
                            arguments: text(&call["function"], "arguments")?,
                        });
                    }
                }
            }
            copy_controls(
                &mut r,
                body,
                &[
                    ("max_tokens", "max_output_tokens"),
                    ("max_completion_tokens", "max_output_tokens"),
                    ("stop", "stop"),
                    ("parallel_tool_calls", "parallel_tool_calls"),
                ],
            );
            if let Some(effort) = body.get("reasoning_effort").filter(|v| !v.is_null()) {
                r.controls
                    .insert("reasoning".into(), json!({"effort":effort}));
            }
            if let Some(c) = body.get("tool_choice") {
                r.controls.insert(
                    "tool_choice".into(),
                    if c.is_object() {
                        json!({"type":"function","name":text(&c["function"],"name")?})
                    } else {
                        c.clone()
                    },
                );
            }
            if let Some(format) = body.get("response_format").filter(|v| !v.is_null()) {
                let f = if format["type"] == "json_schema" {
                    let mut f = format["json_schema"].clone();
                    if !f.is_object() {
                        return Err("Invalid json_schema".into());
                    }
                    f["type"] = json!("json_schema");
                    f
                } else {
                    format.clone()
                };
                r.controls.insert("text".into(), json!({"format":f}));
            }
            if body.get("user").is_some() {
                diagnose(&mut r, "user", "Source identity hint does not transfer");
            }
        }
        Protocol::Responses => {
            check_fields(
                body,
                &[
                    "model",
                    "input",
                    "instructions",
                    "tools",
                    "tool_choice",
                    "stream",
                    "store",
                    "max_output_tokens",
                    "temperature",
                    "top_p",
                    "reasoning",
                    "text",
                    "parallel_tool_calls",
                    "prompt_cache_key",
                    "service_tier",
                    "include",
                ],
            )?;
            if let Some(instructions) = body.get("instructions").filter(|v| !v.is_null()) {
                message(&mut r, "developer", parts(instructions, protocol)?);
            }
            if body["input"].is_string() {
                message(&mut r, "user", parts(&body["input"], protocol)?);
            } else {
                for item in body["input"]
                    .as_array()
                    .ok_or("input must be string or array")?
                {
                    r.items.push(decode_output_item(item)?);
                }
            }
            copy_controls(
                &mut r,
                body,
                &[
                    ("max_output_tokens", "max_output_tokens"),
                    ("reasoning", "reasoning"),
                    ("text", "text"),
                    ("tool_choice", "tool_choice"),
                    ("parallel_tool_calls", "parallel_tool_calls"),
                ],
            );
        }
    }
    if r.items.is_empty() {
        return Err("Request must contain messages".into());
    }
    if let Some(tools) = body.get("tools").filter(|v| !v.is_null()) {
        for tool in tools.as_array().ok_or("tools must be an array")? {
            let t = if protocol == Protocol::ChatCompletions {
                if tool["type"] != "function" {
                    return Err("Native tools require a native protocol path".into());
                }
                &tool["function"]
            } else {
                tool
            };
            if t.get("type").is_some_and(|v| {
                v != "function" && !(protocol == Protocol::Messages && v == "custom")
            }) {
                return Err("Native tools require a native protocol path".into());
            }
            let parameters = t
                .get(if protocol == Protocol::Messages {
                    "input_schema"
                } else {
                    "parameters"
                })
                .cloned()
                .unwrap_or(json!({"type":"object","properties":{}}));
            r.tools.push(Tool {
                name: text(t, "name")?,
                description: t["description"].as_str().map(str::to_owned),
                parameters,
                // Responses may normalize omitted strictness into strict mode,
                // making optional arguments required. Preserve source defaults.
                strict: t["strict"]
                    .as_bool()
                    .or_else(|| (protocol != Protocol::Responses).then_some(false)),
            });
            if t.get("cache_control").is_some() {
                diagnose(
                    &mut r,
                    "tools.cache_control",
                    "Source cache breakpoints do not transfer",
                );
            }
        }
    }
    Ok(r)
}

pub fn decode_output_item(item: &Value) -> Result<Item, String> {
    Ok(match item["type"].as_str().unwrap_or("message") {
        "message" => match parts(&item["content"], Protocol::Responses) {
            Ok(content) => Item::Message {
                role: text(item, "role")?,
                content,
            },
            Err(_) => Item::Native {
                opaque: Opaque {
                    protocol: Protocol::Responses,
                    value: item.clone(),
                },
            },
        },
        "function_call" => Item::ToolCall {
            id: text(item, "call_id")?,
            name: text(item, "name")?,
            arguments: text(item, "arguments")?,
        },
        "function_call_output" => Item::ToolResult {
            id: text(item, "call_id")?,
            content: parts(&item["output"], Protocol::Responses)?,
            is_error: false,
        },
        "reasoning" => Item::Reasoning {
            summary: item["summary"]
                .as_array()
                .map(|a| {
                    a.iter()
                        .filter_map(|p| p["text"].as_str())
                        .collect::<Vec<_>>()
                        .join("\n\n")
                })
                .unwrap_or_default(),
            opaque: Some(Opaque {
                protocol: Protocol::Responses,
                value: item.clone(),
            }),
        },
        _ => Item::Native {
            opaque: Opaque {
                protocol: Protocol::Responses,
                value: item.clone(),
            },
        },
    })
}
fn encode_parts(parts: &[Content], protocol: Protocol, role: &str) -> Result<Value, String> {
    Ok(Value::Array(parts.iter().map(|part|Ok(match part {
        Content::Text{text}=>json!({"type":if protocol==Protocol::Responses {if role=="assistant"{"output_text"}else{"input_text"}}else{"text"},"text":text}),
        Content::Refusal{text}=>if protocol==Protocol::Responses{json!({"type":"refusal","refusal":text})}else{json!({"type":"text","text":text})},
        Content::Image{url,detail}=>match protocol {
            Protocol::Responses=>json!({"type":"input_image","image_url":url,"detail":detail.as_deref().unwrap_or("auto")}),
            Protocol::ChatCompletions=>json!({"type":"image_url","image_url":{"url":url,"detail":detail.as_deref().unwrap_or("auto")}}),
            Protocol::Messages=>if let Some((media,
data))=url.strip_prefix("data:").and_then(|s|s.split_once(";base64,")){
json!({
"type":"image",
"source":{
"type":"base64",
"media_type":media,
"data":data}
}
)}
else{
json!({
"type":"image",
"source":{
"type":"url",
"url":url}
}
)}
,

        },
        Content::File{data,filename}=>match protocol {
            Protocol::Responses=>json!({"type":"input_file","file_data":data,"filename":filename.as_deref().unwrap_or("document.pdf")}),
            Protocol::Messages=>{
let (media,
data)=data.strip_prefix("data:").and_then(|s|s.split_once(";base64,")).ok_or("Only inline files can be converted to Messages")?;
json!({
"type":"document",
"source":{
"type":"base64",
"media_type":media,
"data":data}
}
)}
,

            Protocol::ChatCompletions=>return Err("File conversion to Chat Completions is not supported".into()),
        },
    })).collect::<Result<Vec<_>,String>>()?))
}
pub fn encode_request(r: &Request, protocol: Protocol) -> Result<Translation, String> {
    let mut out = json!({"model":r.model});
    let mut diagnostics = r.diagnostics.clone();
    let mut input = Vec::new();
    let mut system = Vec::new();
    for item in &r.items {
        match item {
            Item::Message{role,content}=> {
                if protocol == Protocol::Messages && content.iter().any(|p| matches!(p, Content::Image { detail: Some(_), .. })) {
                    diagnostics.push(Diagnostic { field: "image.detail".into(), reason: "Image detail has no target equivalent".into() });
                }
                let parts=encode_parts(content,protocol,role)?;
                if protocol==Protocol::Messages && (role=="system"||role=="developer") {system.extend(parts.as_array().cloned().unwrap_or_default());}
                else {input.push(json!({"role":role,"content":parts}));}
            },
            Item::ToolCall{id,name,arguments}=>input.push(match protocol {
                Protocol::Responses=>json!({"type":"function_call","call_id":id,"name":name,"arguments":arguments}),
                Protocol::Messages=>json!({
"role":"assistant",
"content":[{
"type":"tool_use",
"id":id,
"name":name,
"input":serde_json::from_str::<Value>(arguments).map_err(|_|"Invalid completed tool arguments")?}
]}
),

                Protocol::ChatCompletions=>json!({"role":"assistant","content":null,"tool_calls":[{"id":id,"type":"function","function":{"name":name,"arguments":arguments}}]}),
            }),
            Item::ToolResult{id,content,is_error}=> {
                let parts=encode_parts(content,protocol,"user")?;
                let output=if content.iter().all(|p|matches!(p,
Content::Text{
..}
)) {
json!(content.iter().filter_map(|p|if let Content::Text{
text}
=p{
Some(text.as_str())}
else{
None}
).collect::<Vec<_>>().join("\n"))}
else{
parts}
;

                if *is_error && protocol!=Protocol::Messages {
diagnostics.push(Diagnostic{
field:"tool_result.is_error".into(),
reason:"Error marker has no target equivalent; result content retained".into()}
);
}

                input.push(match protocol {
Protocol::Responses=>json!({
"type":"function_call_output",
"call_id":id,
"output":output}
),
Protocol::Messages=>json!({
"role":"user",
"content":[{
"type":"tool_result",
"tool_use_id":id,
"content":output,
"is_error":is_error}
]}
),
Protocol::ChatCompletions=>json!({
"role":"tool",
"tool_call_id":id,
"content":output}
)}
);

            },
            Item::Reasoning{summary,opaque}=>match (protocol,opaque) {
                (Protocol::Responses,Some(o)) if o.protocol==Protocol::Responses=>input.push(o.value.clone()),
                (Protocol::Messages,
Some(o))=>input.push(json!({
"role":"assistant",
"content":[if o.protocol==Protocol::Messages{
o.value.clone()}
else{
json!({
"type":"thinking",
"thinking":summary,
"signature":reasoning_signature(&o.value)}
)}
]}
)),

                (_,Some(_))=>return Err("Opaque reasoning cannot be replayed to this protocol/provider".into()),
                _=>{if !summary.is_empty(){return Err("Unsigned reasoning cannot be replayed across protocols".into());}},
            },
            Item::Native{opaque}=>if opaque.protocol==protocol{input.push(opaque.value.clone())}else{return Err("Native item cannot be translated to this protocol".into());},
        }
    }
    if protocol == Protocol::Messages {
        // Messages requires adjacent tool calls/results to remain within their respective turn.
        let mut merged: Vec<Value> = vec![];
        for m in input {
            if let Some(last) = merged.last_mut().filter(|l| l["role"] == m["role"]) {
                if let (Some(dst), Some(src)) = (
                    last.get_mut("content").and_then(Value::as_array_mut),
                    m["content"].as_array(),
                ) {
                    dst.extend(src.iter().cloned());
                }
            } else {
                merged.push(m);
            }
        }
        out["messages"] = json!(merged);
        if !system.is_empty() {
            out["system"] = json!(system);
        }
    } else {
        out[if protocol == Protocol::Responses {
            "input"
        } else {
            "messages"
        }] = json!(input);
    }
    for (key, v) in &r.controls {
        match (protocol, key.as_str()) {
            (Protocol::Responses, _) => out[key] = v.clone(),
            (Protocol::Messages, "max_output_tokens") => out["max_tokens"] = v.clone(),
            (Protocol::Messages, "stop") => out["stop_sequences"] = v.clone(),
            (Protocol::Messages, "reasoning") => {
                out["thinking"] = if v["effort"] == "none" {
                    json!({"type":"disabled"})
                } else {
                    json!({"type":"adaptive"})
                };
                if v["effort"] != "none" {
                    out["output_config"]["effort"] = v["effort"].clone();
                }
            }
            (Protocol::Messages, "text") => {
                let mut f = v["format"].clone();
                if let Some(o) = f.as_object_mut() {
                    o.remove("name");
                    o.remove("strict");
                }
                out["output_config"]["format"] = f;
            }
            (Protocol::Messages, "tool_choice") => {
                out["tool_choice"] = if v.is_object() {
                    json!({"type":"tool","name":v["name"]})
                } else {
                    json!({"type":match v.as_str(){Some("required")=>"any",Some("none")=>"none",_=>"auto"}})
                }
            }
            (Protocol::Messages, "parallel_tool_calls") => {}
            (Protocol::ChatCompletions, "max_output_tokens") => {
                out["max_completion_tokens"] = v.clone()
            }
            (Protocol::ChatCompletions, "reasoning") => {
                out["reasoning_effort"] = v["effort"].clone()
            }
            (Protocol::ChatCompletions, "text") => {
                out["response_format"] = if v["format"]["type"] == "json_schema" {
                    let mut f = v["format"].clone();
                    f.as_object_mut().ok_or("Invalid format")?.remove("type");
                    json!({"type":"json_schema","json_schema":f})
                } else {
                    v["format"].clone()
                }
            }
            (Protocol::ChatCompletions, "tool_choice") => {
                out["tool_choice"] = if v.is_object() {
                    json!({"type":"function","function":{"name":v["name"]}})
                } else {
                    v.clone()
                }
            }
            (Protocol::ChatCompletions, "store" | "prompt_cache_key" | "service_tier") => {
                out[key] = v.clone()
            }
            (_, "store" | "prompt_cache_key" | "service_tier")
            | (Protocol::ChatCompletions, "top_k") => {
                diagnostics.push(Diagnostic {
                    field: key.clone(),
                    reason: "Control unavailable in target protocol".into(),
                });
            }
            _ => out[key] = v.clone(),
        }
    }
    if protocol == Protocol::Messages
        && let Some(parallel) = r
            .controls
            .get("parallel_tool_calls")
            .and_then(Value::as_bool)
    {
        if out.get("tool_choice").is_none() {
            out["tool_choice"] = json!({"type":"auto"});
        }
        out["tool_choice"]["disable_parallel_tool_use"] = json!(!parallel);
    }
    if !r.tools.is_empty() {
        out["tools"]=json!(r.tools.iter().map(|t| {
        let mut f=json!({"name":t.name,if protocol==Protocol::Messages{"input_schema"}else{"parameters"}:t.parameters});
        if let Some(d)=&t.description{f["description"]=json!(d);}
        if protocol!=Protocol::Messages{f["type"]=json!("function");if let Some(strict)=t.strict{f["strict"]=json!(strict);}}
        if protocol==Protocol::ChatCompletions {f.as_object_mut().unwrap().remove("type");json!({"type":"function","function":f})}else{f}
    }).collect::<Vec<_>>());
    }
    Ok(Translation {
        body: out,
        diagnostics,
    })
}
pub fn translate_request(
    source: Protocol,
    target: Protocol,
    body: &Value,
) -> Result<Translation, String> {
    if source == target {
        if !body.is_object() {
            return Err("Expected object".into());
        }
        return Ok(Translation {
            body: body.clone(),
            diagnostics: vec![],
        });
    }
    encode_request(&decode_request(source, body)?, target)
}
