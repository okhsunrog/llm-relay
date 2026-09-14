use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Protocol {
    Messages,
    ChatCompletions,
    Responses,
}

/// A provider-bound payload. Never reinterpret another protocol's opaque state.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Opaque {
    pub protocol: Protocol,
    pub value: Value,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Content {
    Text {
        text: String,
    },
    Image {
        url: String,
        detail: Option<String>,
    },
    File {
        data: String,
        filename: Option<String>,
    },
    Refusal {
        text: String,
    },
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Item {
    Message {
        role: String,
        content: Vec<Content>,
    },
    ToolCall {
        id: String,
        name: String,
        arguments: String,
    },
    ToolResult {
        id: String,
        content: Vec<Content>,
        is_error: bool,
    },
    Reasoning {
        summary: String,
        opaque: Option<Opaque>,
    },
    Native {
        opaque: Opaque,
    },
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Tool {
    pub name: String,
    pub description: Option<String>,
    pub parameters: Value,
    pub strict: Option<bool>,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Diagnostic {
    pub field: String,
    pub reason: String,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Request {
    pub model: String,
    pub items: Vec<Item>,
    pub tools: Vec<Tool>,
    /// Common controls use Responses spellings, but have no backend policy.
    pub controls: Map<String, Value>,
    pub diagnostics: Vec<Diagnostic>,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Translation {
    pub body: Value,
    pub diagnostics: Vec<Diagnostic>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Policy {
    Strict,
    Compatible,
}
impl Translation {
    pub fn enforce(self, policy: Policy) -> Result<Self, String> {
        if policy == Policy::Strict && !self.diagnostics.is_empty() {
            return Err(format!(
                "Conversion would change: {}",
                self.diagnostics
                    .iter()
                    .map(|d| d.field.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
        }
        Ok(self)
    }
}
