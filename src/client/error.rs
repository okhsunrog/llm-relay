use thiserror::Error;

#[derive(Debug, Error)]
pub enum LlmError {
    #[error("Invalid client configuration: {0}")]
    Config(String),
    #[error("HTTP client error: {0}")]
    Client(String),

    #[error("Request failed: {0}")]
    Request(#[from] reqwest::Error),

    #[error("API error ({status}): {body}")]
    ApiError { status: u16, body: String },

    #[error("Failed to parse response: {0}")]
    ParseResponse(String),

    #[error("Structured response did not match the requested schema: {error}; body: {body}")]
    InvalidStructuredOutput { error: String, body: String },

    #[error("Empty response from API")]
    EmptyResponse,

    #[error("Conversion error: {0}")]
    Conversion(String),

    #[error("Response exceeded {limit} bytes (received {actual})")]
    ResponseTooLarge { limit: usize, actual: usize },

    #[error("Streaming protocol error: {0}")]
    Stream(String),
}

impl LlmError {
    /// Whether the same request can be retried without changing its payload.
    ///
    /// Transport failures and provider throttling or server errors are
    /// transient. Schema, parsing, configuration, and authentication failures
    /// need bounded retries or operator intervention instead.
    pub fn is_transient(&self) -> bool {
        match self {
            Self::Request(_) => true,
            Self::ApiError { status, .. } => {
                matches!(*status, 408 | 429) || (500..=599).contains(status)
            }
            _ => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::LlmError;

    #[test]
    fn classifies_retryable_api_statuses() {
        for status in [408, 429, 500, 529, 599] {
            assert!(
                LlmError::ApiError {
                    status,
                    body: String::new()
                }
                .is_transient(),
                "{status} should be transient"
            );
        }
        for status in [400, 401, 403, 404, 422] {
            assert!(
                !LlmError::ApiError {
                    status,
                    body: String::new()
                }
                .is_transient(),
                "{status} should not be transient"
            );
        }
    }
}
