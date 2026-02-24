use crate::types::anthropic::{OutputConfig, ThinkingParam};
use crate::types::common::{EffortLevel, ThinkingConfig};

/// Build Anthropic extended thinking API parameters from a ThinkingConfig.
///
/// Returns `(thinking_param, output_config)` for inclusion in a MessagesRequest.
/// For adaptive thinking, "high" effort is the default and omits `output_config`.
pub fn build_thinking_params(
    config: Option<&ThinkingConfig>,
) -> (Option<ThinkingParam>, Option<OutputConfig>) {
    match config {
        Some(ThinkingConfig::Adaptive { effort }) => {
            let output_config = if *effort == EffortLevel::High {
                None
            } else {
                Some(OutputConfig {
                    effort: effort.as_str().to_string(),
                })
            };
            (Some(ThinkingParam::Adaptive), output_config)
        }
        Some(ThinkingConfig::Enabled { budget_tokens }) => (
            Some(ThinkingParam::Enabled {
                budget_tokens: *budget_tokens,
            }),
            None,
        ),
        None => (None, None),
    }
}

/// Parse a model suffix like `"claude-sonnet-4-5(medium)"` into `(base_model, effort_string)`.
///
/// Returns the base model and optional effort if the suffix is valid.
/// Valid suffixes: named efforts (none, off, disabled, low, minimal, medium, med, high, xhigh, max, auto)
/// or numeric values (for budget_tokens on older models).
pub fn parse_model_suffix(model: &str) -> (String, Option<String>) {
    let Some(open_paren) = model.rfind('(') else {
        return (model.to_string(), None);
    };

    if !model.ends_with(')') {
        return (model.to_string(), None);
    }

    let base_model = &model[..open_paren];
    let suffix = &model[open_paren + 1..model.len() - 1];

    let is_valid = matches!(
        suffix.to_lowercase().as_str(),
        "none"
            | "off"
            | "disabled"
            | "low"
            | "minimal"
            | "medium"
            | "med"
            | "high"
            | "xhigh"
            | "max"
            | "auto"
    ) || suffix.parse::<u32>().is_ok();

    if is_valid {
        (base_model.to_string(), Some(suffix.to_string()))
    } else {
        (model.to_string(), None)
    }
}

/// Check if a model supports adaptive thinking.
///
/// Adaptive thinking is supported on Claude Opus 4.6 and Claude Sonnet 4.6.
/// Older models (Sonnet 4.5, etc.) use manual extended thinking with `budget_tokens` instead.
pub fn supports_adaptive_thinking(model: &str) -> bool {
    let lower = model.to_lowercase();
    lower.contains("opus-4-6")
        || lower.contains("sonnet-4-6")
        || lower.starts_with("claude-opus-4-6")
        || lower.starts_with("claude-sonnet-4-6")
}

/// Build a ThinkingConfig based on model name and effort string.
///
/// For models supporting adaptive thinking (Opus 4.6, Sonnet 4.6): uses adaptive mode with effort levels.
/// For older models: uses manual extended thinking with budget_tokens.
///
/// Returns `None` if thinking is disabled (none/off/disabled/0).
pub fn build_thinking_for_model(model: &str, effort: &str) -> Option<ThinkingConfig> {
    let effort_lower = effort.to_lowercase();

    // Check for disabled
    if matches!(effort_lower.as_str(), "none" | "off" | "disabled") {
        return None;
    }

    if supports_adaptive_thinking(model) {
        // Adaptive thinking with effort levels (Opus 4.6, Sonnet 4.6)
        let level = match effort_lower.as_str() {
            "low" | "minimal" => EffortLevel::Low,
            "medium" | "med" | "auto" => EffortLevel::Medium,
            "high" => EffortLevel::High,
            "xhigh" | "max" => EffortLevel::Max,
            _ => {
                if let Ok(n) = effort.parse::<u32>() {
                    match n {
                        0 => return None,
                        1..=2048 => EffortLevel::Low,
                        2049..=16384 => EffortLevel::Medium,
                        16385..=49152 => EffortLevel::High,
                        _ => EffortLevel::Max,
                    }
                } else {
                    EffortLevel::High
                }
            }
        };
        Some(ThinkingConfig::Adaptive { effort: level })
    } else {
        // Older models: manual extended thinking with budget_tokens
        let budget_tokens = match effort_lower.as_str() {
            "low" | "minimal" => 1024,
            "medium" | "med" => 8192,
            "high" => 32000,
            "xhigh" | "max" => 64000,
            "auto" => 16000,
            _ => effort.parse::<u32>().unwrap_or(8192),
        };
        Some(ThinkingConfig::Enabled { budget_tokens })
    }
}
