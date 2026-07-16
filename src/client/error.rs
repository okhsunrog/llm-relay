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
