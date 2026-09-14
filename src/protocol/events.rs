//! One Responses decoder feeds native passthrough, both compatibility encoders,
//! and the JSON collector. The caller owns I/O, timeout and cancellation policy.
use super::{
    model::*,
    request::{decode_output_item, encode_request, reasoning_signature},
    responses,
};
use serde_json::{Value, json};
use std::collections::{BTreeMap, HashMap};

#[derive(Debug, Clone)]
pub struct Completion {
    pub id: String,
    pub model: String,
    pub created: u64,
    pub items: Vec<Item>,
    pub usage: Option<crate::Usage>,
    pub stop_reason: String,
    pub native: Opaque,
}
impl Completion {
    pub fn render(&self, protocol: Protocol) -> Result<Value, String> {
        if protocol == self.native.protocol {
            return Ok(self.native.value.clone());
        }
        if self
            .items
            .iter()
            .any(|item| matches!(item, Item::Native { .. }))
        {
            return Err("Native output item cannot be represented in target protocol".into());
        }
        if protocol == Protocol::ChatCompletions {
            return Ok(responses::to_chat(&self.native.value));
        }
        let r = Request {
            model: self.model.clone(),
            items: self.items.clone(),
            tools: vec![],
            controls: Default::default(),
            diagnostics: vec![],
        };
        let translated = encode_request(&r, Protocol::Messages)?;
        let content = translated.body["messages"]
            .as_array()
            .into_iter()
            .flatten()
            .flat_map(|m| m["content"].as_array().into_iter().flatten().cloned())
            .collect::<Vec<_>>();
        Ok(json!({
        "id":self.id,
        "type":"message",
        "role":"assistant",
        "model":self.model,
        "content":content,
        "stop_reason":self.stop_reason,
        "stop_sequence":null,
        "usage":self.usage.clone().unwrap_or_default()}
        ))
    }
}
#[derive(Debug, Clone)]
pub enum Event {
    Start {
        id: String,
        model: String,
        created: u64,
    },
    ItemStart {
        index: u64,
        item: Item,
    },
    Text {
        item: u64,
        part: u64,
        text: String,
    },
    Refusal {
        item: u64,
        part: u64,
        text: String,
    },
    Reasoning {
        item: u64,
        text: String,
    },
    Arguments {
        item: u64,
        text: String,
    },
    ItemEnd {
        index: u64,
        item: Item,
    },
    Finish(Completion),
    Other,
}
#[derive(Debug, Clone)]
pub struct Decoded {
    pub native: Value,
    pub event: Event,
    pub usage: Option<crate::Usage>,
}
#[derive(Default)]
pub struct Decoder {
    accumulator: responses::Accumulator,
    finished: bool,
    usage: Option<crate::Usage>,
}
impl Decoder {
    pub fn usage(&self) -> Option<&crate::Usage> {
        self.usage.as_ref()
    }
    pub fn finish(&self) -> Result<(), String> {
        if self.finished {
            Ok(())
        } else {
            Err("Upstream stream ended before completion".into())
        }
    }
    pub fn decode(&mut self, mut raw: Value) -> Result<Decoded, String> {
        if self.finished {
            return Err("Event after terminal response".into());
        }
        self.accumulator.observe(&mut raw);
        if let Some(u) = responses::usage(&raw["response"]) {
            self.usage = Some(u);
        }
        let index = || {
            raw["output_index"]
                .as_u64()
                .ok_or_else(|| "Missing output_index".to_owned())
        };
        let delta = || {
            raw["delta"]
                .as_str()
                .map(str::to_owned)
                .ok_or_else(|| "Missing delta".to_owned())
        };
        let event = match raw["type"].as_str().ok_or("Missing event type")? {
            "response.created" => Event::Start {
                id: raw["response"]["id"]
                    .as_str()
                    .ok_or("Missing response id")?
                    .into(),
                model: raw["response"]["model"].as_str().unwrap_or("").into(),
                created: raw["response"]["created_at"].as_u64().unwrap_or(0),
            },
            "response.output_item.added" => Event::ItemStart {
                index: index()?,
                item: decode_output_item(&raw["item"])?,
            },
            "response.output_item.done" => Event::ItemEnd {
                index: index()?,
                item: decode_output_item(&raw["item"])?,
            },
            "response.output_text.delta" => Event::Text {
                item: index()?,
                part: raw["content_index"]
                    .as_u64()
                    .ok_or("Missing content_index")?,
                text: delta()?,
            },
            "response.refusal.delta" => Event::Refusal {
                item: index()?,
                part: raw["content_index"]
                    .as_u64()
                    .ok_or("Missing content_index")?,
                text: delta()?,
            },
            "response.reasoning_summary_text.delta" => Event::Reasoning {
                item: index()?,
                text: delta()?,
            },
            "response.function_call_arguments.delta" => Event::Arguments {
                item: index()?,
                text: delta()?,
            },
            "response.completed" | "response.incomplete" => {
                let response = &raw["response"];
                let items = response["output"]
                    .as_array()
                    .ok_or("Missing response output")?
                    .iter()
                    .map(decode_output_item)
                    .collect::<Result<Vec<_>, _>>()?;
                let tool = items.iter().any(|i| matches!(i, Item::ToolCall { .. }));
                let stop = if response["incomplete_details"]["reason"] == "content_filter" {
                    "refusal"
                } else if response["status"] == "incomplete" {
                    "max_tokens"
                } else if tool {
                    "tool_use"
                } else {
                    "end_turn"
                };
                let completion = Completion {
                    id: response["id"].as_str().unwrap_or("").into(),
                    model: response["model"].as_str().unwrap_or("").into(),
                    created: response["created_at"].as_u64().unwrap_or(0),
                    items,
                    usage: self.usage.clone(),
                    stop_reason: stop.into(),
                    native: Opaque {
                        protocol: Protocol::Responses,
                        value: response.clone(),
                    },
                };
                self.finished = true;
                Event::Finish(completion)
            }
            "error" | "response.failed" => {
                self.finished = true;
                return Err("Upstream generation failed".into());
            }
            _ => Event::Other,
        };
        Ok(Decoded {
            native: raw,
            event,
            usage: self.usage.clone(),
        })
    }
}
#[derive(Debug, Clone)]
pub struct Frame {
    pub event: Option<String>,
    pub data: Value,
}
impl Frame {
    pub fn sse(&self) -> String {
        match &self.event {
            Some(name) => format!("event: {name}\ndata: {}\n\n", self.data),
            None => format!("data: {}\n\n", self.data),
        }
    }
}
struct Block {
    index: usize,
    text: String,
    closed: bool,
}
pub struct Encoder {
    protocol: Protocol,
    model: String,
    id: String,
    created: u64,
    include_usage: bool,
    blocks: BTreeMap<(u64, u64), Block>,
    calls: HashMap<u64, usize>,
}
impl Encoder {
    pub fn needs_done_sentinel(&self) -> bool {
        self.protocol == Protocol::ChatCompletions
    }

    pub fn new(protocol: Protocol, model: String, include_usage: bool) -> Self {
        Self {
            protocol,
            model,
            id: String::new(),
            created: 0,
            include_usage,
            blocks: Default::default(),
            calls: Default::default(),
        }
    }
    pub fn error(&self, message: &str) -> Frame {
        let data = if self.protocol == Protocol::Messages {
            json!({"type":"error","error":{"type":"api_error","message":message}})
        } else {
            json!({"type":"error","error":{"type":"upstream_error","message":message}})
        };
        Frame {
            event: Some("error".into()),
            data,
        }
    }
    fn frame(data: Value) -> Frame {
        Frame {
            event: data["type"].as_str().map(str::to_owned),
            data,
        }
    }
    fn chat(&self, delta: Value, finish: Value) -> Frame {
        Frame {
            event: None,
            data: json!({"id":self.id,"object":"chat.completion.chunk","model":self.model,"created":self.created,"choices":[{"index":0,"delta":delta,"finish_reason":finish}]}),
        }
    }
    fn block(&mut self, key: (u64, u64), content: Value, frames: &mut Vec<Frame>) -> usize {
        let next = self.blocks.len();
        self.blocks
            .entry(key)
            .or_insert_with(|| {
                frames.push(Self::frame(
                    json!({"type":"content_block_start","index":next,"content_block":content}),
                ));
                Block {
                    index: next,
                    text: String::new(),
                    closed: false,
                }
            })
            .index
    }
    fn close_item(&mut self, item: u64, frames: &mut Vec<Frame>) {
        for ((id, _), block) in &mut self.blocks {
            if *id == item && !block.closed {
                frames.push(Self::frame(
                    json!({"type":"content_block_stop","index":block.index}),
                ));
                block.closed = true;
            }
        }
    }
    pub fn encode(&mut self, decoded: &Decoded) -> Result<Vec<Frame>, String> {
        if self.protocol == Protocol::Responses {
            return Ok(vec![Self::frame(decoded.native.clone())]);
        }
        if matches!(
            &decoded.event,
            Event::ItemStart {
                item: Item::Native { .. },
                ..
            } | Event::ItemEnd {
                item: Item::Native { .. },
                ..
            }
        ) || matches!(&decoded.event, Event::Finish(c) if c.items.iter().any(|i| matches!(i, Item::Native { .. })))
        {
            return Err("Native output cannot be represented in target protocol".into());
        }
        let mut frames = vec![];
        if let Event::Start { id, model, created } = &decoded.event {
            self.id = id.clone();
            self.created = *created;
            if !model.is_empty() {
                self.model = model.clone();
            }
        }
        if self.protocol == Protocol::ChatCompletions {
            match &decoded.event {
                Event::Start { .. } => {
                    frames.push(self.chat(json!({"role":"assistant","content":""}), Value::Null))
                }
                Event::Text { text, .. } => {
                    frames.push(self.chat(json!({"content":text}), Value::Null))
                }
                Event::Refusal { text, .. } => {
                    frames.push(self.chat(json!({"refusal":text}), Value::Null))
                }
                Event::Reasoning { text, .. } => {
                    frames.push(self.chat(json!({"reasoning_content":text}), Value::Null))
                }
                Event::ItemStart {
                    index,
                    item:
                        Item::ToolCall {
                            id,
                            name,
                            arguments,
                        },
                } => {
                    let n = self.calls.len();
                    self.calls.insert(*index, n);
                    frames.push(self.chat(json!({"tool_calls":[{"index":n,"id":id,"type":"function","function":{"name":name,"arguments":arguments}}]}),Value::Null));
                }
                Event::Arguments { item, text } => {
                    let n = self
                        .calls
                        .get(item)
                        .ok_or("Tool argument delta before tool start")?;
                    frames.push(self.chat(
                        json!({"tool_calls":[{"index":n,"function":{"arguments":text}}]}),
                        Value::Null,
                    ));
                }
                Event::Finish(c) => {
                    let reason = match c.stop_reason.as_str() {
                        "max_tokens" => "length",
                        "refusal" => "content_filter",
                        "tool_use" => "tool_calls",
                        _ => "stop",
                    };
                    frames.push(self.chat(json!({}), json!(reason)));
                    if self.include_usage {
                        let mut f = self.chat(json!({}), Value::Null);
                        f.data["choices"] = json!([]);
                        f.data["usage"] = responses::to_chat(&c.native.value)["usage"].clone();
                        frames.push(f);
                    }
                }
                Event::ItemStart {
                    item: Item::Native { .. },
                    ..
                } => return Err("Native output cannot be represented in Chat Completions".into()),
                _ => {}
            }
            return Ok(frames);
        }
        match &decoded.event {
            Event::Start { .. } => frames.push(Self::frame(json!({
            "type":"message_start",
            "message":{
            "id":self.id,
            "type":"message",
            "role":"assistant",
            "model":self.model,
            "content":[],
            "stop_reason":null,
            "stop_sequence":null,
            "usage":{
            "input_tokens":0,
            "output_tokens":0}
            }
            }
            ))),

            Event::ItemStart {
                index,
                item:
                    Item::ToolCall {
                        id,
                        name,
                        arguments,
                    },
            } => {
                let n = self.block(
                    (*index, 0),
                    json!({
                    "type":"tool_use",
                    "id":id,
                    "name":name,
                    "input":{
                    }
                    }
                    ),
                    &mut frames,
                );
                if !arguments.is_empty() {
                    frames.push(Self::frame(json!({
                    "type":"content_block_delta",
                    "index":n,
                    "delta":{
                    "type":"input_json_delta",
                    "partial_json":arguments}
                    }
                    )));
                    if let Some(b) = self.blocks.get_mut(&(*index, 0)) {
                        b.text = arguments.clone();
                    }
                }
            }
            Event::ItemStart {
                index,
                item: Item::Reasoning { .. },
            } => {
                self.block(
                    (*index, 0),
                    json!({"type":"thinking","thinking":""}),
                    &mut frames,
                );
            }
            Event::Text { item, part, text } | Event::Refusal { item, part, text } => {
                let n = self.block(
                    (*item, *part),
                    json!({
                    "type":"text",
                    "text":""}
                    ),
                    &mut frames,
                );
                frames.push(Self::frame(json!({
                "type":"content_block_delta",
                "index":n,
                "delta":{
                "type":"text_delta",
                "text":text}
                }
                )));
                if let Some(b) = self.blocks.get_mut(&(*item, *part)) {
                    b.text.push_str(text);
                }
            }
            Event::Reasoning { item, text } => {
                let n = self.block(
                    (*item, 0),
                    json!({
                    "type":"thinking",
                    "thinking":""}
                    ),
                    &mut frames,
                );
                frames.push(Self::frame(json!({
                "type":"content_block_delta",
                "index":n,
                "delta":{
                "type":"thinking_delta",
                "thinking":text}
                }
                )));
            }
            Event::Arguments { item, text } => {
                let b = self
                    .blocks
                    .get_mut(&(*item, 0))
                    .ok_or("Tool delta before tool start")?;
                if b.closed {
                    return Err("Delta after block end".into());
                }
                b.text.push_str(text);
                frames.push(Self::frame(json!({
                "type":"content_block_delta",
                "index":b.index,
                "delta":{
                "type":"input_json_delta",
                "partial_json":text}
                }
                )));
            }
            Event::ItemEnd { index, item } => {
                match item {
                    Item::Reasoning {
                        summary,
                        opaque: Some(o),
                    } => {
                        let n = self.block(
                            (*index, 0),
                            json!({
                            "type":"thinking",
                            "thinking":""}
                            ),
                            &mut frames,
                        );
                        let _ = summary;
                        frames.push(Self::frame(json!({
                        "type":"content_block_delta",
                        "index":n,
                        "delta":{
                        "type":"signature_delta",
                        "signature":reasoning_signature(&o.value)}
                        }
                        )));
                    }
                    Item::ToolCall { arguments, .. } => {
                        let _: Value = serde_json::from_str(arguments)
                            .map_err(|_| "Invalid completed tool arguments")?;
                        let b = self
                            .blocks
                            .get_mut(&(*index, 0))
                            .ok_or("Tool end before tool start")?;
                        if b.text != *arguments {
                            let remaining = arguments
                                .strip_prefix(&b.text)
                                .ok_or("Tool arguments differ from streamed deltas")?;
                            frames.push(Self::frame(json!({
                            "type":"content_block_delta",
                            "index":b.index,
                            "delta":{
                            "type":"input_json_delta",
                            "partial_json":remaining}
                            }
                            )));
                            b.text = arguments.clone();
                        }
                    }
                    Item::Native { .. } => {
                        return Err("Native output cannot be represented in Messages".into());
                    }
                    _ => {}
                }
                self.close_item(*index, &mut frames);
            }
            Event::Finish(c) => {
                let ids = self.blocks.keys().map(|(id, _)| *id).collect::<Vec<_>>();
                for id in ids {
                    self.close_item(id, &mut frames);
                }
                frames.push(Self::frame(json!({"type":"message_delta","delta":{"stop_reason":c.stop_reason,"stop_sequence":null},"usage":c.usage.clone().unwrap_or_default()})));
                frames.push(Self::frame(json!({"type":"message_stop"})));
            }
            Event::ItemStart {
                item: Item::Native { .. },
                ..
            } => return Err("Native output cannot be represented in Messages".into()),
            _ => {}
        }
        Ok(frames)
    }
}
