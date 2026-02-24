use thiserror::Error;

#[derive(Debug, Error)]
pub enum LlmError {
    #[error("HTTP client error: {0}")]
    Client(String),

    #[error("Request failed: {0}")]
    Request(#[from] reqwest::Error),

    #[error("API error ({status}): {body}")]
    ApiError { status: u16, body: String },

    #[error("Failed to parse response: {0}")]
    ParseResponse(String),

    #[error("Empty response from API")]
    EmptyResponse,

    #[error("Conversion error: {0}")]
    Conversion(String),
}
