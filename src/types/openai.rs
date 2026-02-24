use serde::{Deserialize, Serialize};

use super::common::ResponseFormat;

// ============ Outbound request types ============

/// OpenAI chat completion request.
#[derive(Debug, Serialize)]
pub struct ChatRequest {
    pub model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    pub messages: Vec<ChatMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Tool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<ResponseFormat>,
}

/// OpenAI chat message (for requests).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCallOut>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

impl ChatMessage {
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: "system".to_string(),
            content: Some(content.into()),
            tool_calls: None,
            tool_call_id: None,
        }
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: "user".to_string(),
            content: Some(content.into()),
            tool_calls: None,
            tool_call_id: None,
        }
    }

    pub fn assistant_text(content: impl Into<String>) -> Self {
        Self {
            role: "assistant".to_string(),
            content: Some(content.into()),
            tool_calls: None,
            tool_call_id: None,
        }
    }

    pub fn tool_result(tool_call_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: "tool".to_string(),
            content: Some(content.into()),
            tool_calls: None,
            tool_call_id: Some(tool_call_id.into()),
        }
    }
}

/// OpenAI tool definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tool {
    #[serde(rename = "type")]
    pub tool_type: String,
    pub function: ToolFunction,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolFunction {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
}

/// Outbound tool call in an assistant message.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallOut {
    pub id: String,
    #[serde(rename = "type")]
    pub call_type: String,
    pub function: ToolCallFunction,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallFunction {
    pub name: String,
    pub arguments: String,
}

// ============ Response types ============

/// OpenAI chat completion response.
#[derive(Debug, Serialize, Deserialize)]
pub struct ChatResponse {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub object: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub created: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    pub choices: Vec<Choice>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<ResponseUsage>,
}

impl ChatResponse {
    /// Get the text content of the first choice.
    pub fn text(&self) -> Option<&str> {
        self.choices
            .first()
            .and_then(|c| c.message.content.as_deref())
    }

    /// Get the text content or an error.
    pub fn text_or_err(&self) -> Result<&str, &'static str> {
        self.text().ok_or("No response content (empty choices)")
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Choice {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub index: Option<u32>,
    pub message: ResponseMessage,
    #[serde(default)]
    pub finish_reason: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ResponseMessage {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    pub content: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ResponseToolCall>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseToolCall {
    pub id: String,
    #[serde(default, skip_serializing_if = "Option::is_none", rename = "type")]
    pub call_type: Option<String>,
    pub function: ResponseToolCallFunction,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseToolCallFunction {
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ResponseUsage {
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub total_tokens: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_creation_input_tokens: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_read_input_tokens: Option<u64>,
}

// ============ Embeddings types ============

/// OpenAI-compatible embeddings request.
#[derive(Debug, Serialize)]
pub struct EmbeddingsRequest {
    pub model: String,
    pub input: Vec<String>,
}

/// OpenAI-compatible embeddings response.
#[derive(Debug, Deserialize)]
pub struct EmbeddingsResponse {
    pub data: Vec<EmbeddingObject>,
}

#[derive(Debug, Deserialize)]
pub struct EmbeddingObject {
    pub embedding: Vec<f32>,
    pub index: usize,
}

// ============ Inbound types (for proxy scenarios) ============

/// Inbound OpenAI chat request (for proxy/conversion scenarios).
/// More permissive deserialization than ChatRequest.
#[derive(Debug, Deserialize)]
pub struct InboundChatRequest {
    pub model: Option<String>,
    pub messages: Vec<InboundMessage>,
    #[serde(default)]
    pub max_tokens: Option<u32>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub tools: Option<Vec<serde_json::Value>>,
    #[serde(default)]
    pub reasoning_effort: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct InboundMessage {
    pub role: String,
    pub content: InboundContent,
    #[serde(default)]
    pub tool_calls: Option<Vec<InboundToolCall>>,
    #[serde(default)]
    pub tool_call_id: Option<String>,
}

/// Content that can be a string, array of parts, or null.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum InboundContent {
    Text(String),
    Parts(Vec<InboundContentPart>),
    Null,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
pub enum InboundContentPart {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "image_url")]
    ImageUrl { image_url: ImageUrlData },
}

#[derive(Debug, Deserialize)]
pub struct ImageUrlData {
    pub url: String,
}

#[derive(Debug, Deserialize)]
pub struct InboundToolCall {
    pub id: String,
    pub function: InboundFunction,
}

#[derive(Debug, Deserialize)]
pub struct InboundFunction {
    pub name: String,
    pub arguments: String,
}
