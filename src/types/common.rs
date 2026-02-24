use serde::{Deserialize, Serialize};

/// LLM provider type.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Provider {
    #[default]
    Anthropic,
    #[serde(alias = "openai")]
    OpenAiCompatible,
}

impl Provider {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Anthropic => "anthropic",
            Self::OpenAiCompatible => "openai",
        }
    }

    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "anthropic" => Some(Self::Anthropic),
            "openai" => Some(Self::OpenAiCompatible),
            _ => None,
        }
    }

    pub fn default_base_url(&self) -> &'static str {
        match self {
            Self::Anthropic => "https://api.anthropic.com",
            Self::OpenAiCompatible => "https://api.openai.com",
        }
    }
}

impl std::fmt::Display for Provider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Extended thinking configuration.
///
/// Extended thinking lets Claude think step-by-step before responding.
/// Two modes are available:
/// - **Adaptive**: Claude decides when and how much to think. Controlled via effort level.
///   Supported on Claude Opus 4.6 and Sonnet 4.6.
/// - **Enabled**: Manual extended thinking with an explicit token budget.
///   For older models (Sonnet 4.5, etc.) or when a specific budget is needed.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum ThinkingConfig {
    /// Adaptive thinking — Claude decides when and how much to think.
    Adaptive {
        #[serde(default)]
        effort: EffortLevel,
    },
    /// Manual extended thinking with an explicit token budget.
    Enabled {
        budget_tokens: u32,
    },
}

/// Effort level for adaptive thinking.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum EffortLevel {
    Max,
    #[default]
    High,
    Medium,
    Low,
}

impl EffortLevel {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Max => "max",
            Self::High => "high",
            Self::Medium => "medium",
            Self::Low => "low",
        }
    }

    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "max" => Some(Self::Max),
            "high" => Some(Self::High),
            "medium" | "med" => Some(Self::Medium),
            "low" | "minimal" => Some(Self::Low),
            _ => None,
        }
    }

    pub fn all() -> &'static [EffortLevel] {
        &[Self::Max, Self::High, Self::Medium, Self::Low]
    }
}

impl std::fmt::Display for EffortLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Provider-agnostic tool definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub input_schema: serde_json::Value,
}

/// Token usage information.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Usage {
    pub input_tokens: u32,
    pub output_tokens: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_creation_input_tokens: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_read_input_tokens: Option<u32>,
}

impl Usage {
    pub fn total_tokens(&self) -> u32 {
        self.input_tokens + self.output_tokens
    }
}

/// Response format specification (for JSON mode).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseFormat {
    #[serde(rename = "type")]
    pub format_type: String,
}

impl ResponseFormat {
    pub fn json_object() -> Self {
        Self {
            format_type: "json_object".to_string(),
        }
    }
}

/// Stop reason, normalized across providers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StopReason {
    EndTurn,
    ToolUse,
    MaxTokens,
    Other(String),
}

impl StopReason {
    pub fn from_anthropic(s: &str) -> Self {
        match s {
            "end_turn" => Self::EndTurn,
            "tool_use" => Self::ToolUse,
            "max_tokens" => Self::MaxTokens,
            other => Self::Other(other.to_string()),
        }
    }

    pub fn from_openai(s: &str) -> Self {
        match s {
            "stop" => Self::EndTurn,
            "tool_calls" => Self::ToolUse,
            "length" => Self::MaxTokens,
            other => Self::Other(other.to_string()),
        }
    }

    pub fn to_anthropic(&self) -> &str {
        match self {
            Self::EndTurn => "end_turn",
            Self::ToolUse => "tool_use",
            Self::MaxTokens => "max_tokens",
            Self::Other(s) => s,
        }
    }

    pub fn to_openai(&self) -> &str {
        match self {
            Self::EndTurn => "stop",
            Self::ToolUse => "tool_calls",
            Self::MaxTokens => "length",
            Self::Other(s) => s,
        }
    }

    pub fn is_tool_use(&self) -> bool {
        matches!(self, Self::ToolUse)
    }
}
