use serde::{Deserialize, Serialize};

use super::common::{StopReason, ToolDefinition, Usage};

/// Anthropic content block — the canonical internal representation.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ContentBlock {
    #[serde(rename = "text")]
    Text { text: String },

    #[serde(rename = "thinking")]
    Thinking {
        thinking: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        signature: Option<String>,
    },

    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },

    #[serde(rename = "tool_result")]
    ToolResult {
        tool_use_id: String,
        content: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        is_error: Option<bool>,
    },
}

/// An Anthropic API message.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Message {
    pub role: String,
    pub content: Vec<ContentBlock>,
}

impl Message {
    pub fn user(content: Vec<ContentBlock>) -> Self {
        Self {
            role: "user".to_string(),
            content,
        }
    }

    pub fn assistant(content: Vec<ContentBlock>) -> Self {
        Self {
            role: "assistant".to_string(),
            content,
        }
    }

    pub fn user_text(text: impl Into<String>) -> Self {
        Self::user(vec![ContentBlock::Text { text: text.into() }])
    }

    pub fn tool_results(results: Vec<ContentBlock>) -> Self {
        Self::user(results)
    }
}

/// Extended thinking parameter for the Anthropic API wire format.
///
/// Maps to the `thinking` field in the Messages API request.
/// - `Adaptive`: `{type: "adaptive"}` — Claude decides when/how much to think.
/// - `Enabled`: `{type: "enabled", budget_tokens: N}` — manual extended thinking with a fixed budget.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum ThinkingParam {
    Adaptive,
    Enabled { budget_tokens: u32 },
}

/// Output configuration — controls effort level for adaptive thinking.
#[derive(Debug, Clone, Serialize)]
pub struct OutputConfig {
    pub effort: String,
}

/// Anthropic Messages API request.
#[derive(Debug, Serialize)]
pub struct MessagesRequest {
    pub model: String,
    pub max_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<String>,
    pub messages: Vec<Message>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<ToolDefinition>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking: Option<ThinkingParam>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_config: Option<OutputConfig>,
}

/// Anthropic Messages API response.
#[derive(Debug, Deserialize)]
pub struct MessagesResponse {
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub model: Option<String>,
    pub content: Vec<ContentBlock>,
    pub stop_reason: String,
    #[serde(default)]
    pub usage: Option<Usage>,
}

impl MessagesResponse {
    /// Extract all text content concatenated.
    pub fn text(&self) -> String {
        self.content
            .iter()
            .filter_map(|b| match b {
                ContentBlock::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("")
    }

    /// Extract thinking content concatenated.
    pub fn thinking_text(&self) -> Option<String> {
        let texts: Vec<&str> = self
            .content
            .iter()
            .filter_map(|b| match b {
                ContentBlock::Thinking { thinking, .. } => Some(thinking.as_str()),
                _ => None,
            })
            .collect();
        if texts.is_empty() {
            None
        } else {
            Some(texts.join(""))
        }
    }

    /// Get the stop reason as a normalized StopReason enum.
    pub fn stop(&self) -> StopReason {
        StopReason::from_anthropic(&self.stop_reason)
    }

    /// Check if the model wants to call tools.
    pub fn has_tool_use(&self) -> bool {
        self.stop().is_tool_use()
    }

    /// Extract tool use blocks.
    pub fn tool_uses(&self) -> Vec<&ContentBlock> {
        self.content
            .iter()
            .filter(|b| matches!(b, ContentBlock::ToolUse { .. }))
            .collect()
    }
}
