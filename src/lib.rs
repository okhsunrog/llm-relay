pub mod convert;
pub mod types;

#[cfg(feature = "client")]
pub mod client;

// Re-export commonly used items at crate root
#[cfg(feature = "client")]
pub use client::{ChatOptions, ClientConfig, LlmClient, LlmError};

#[cfg(feature = "embeddings")]
pub use client::{EmbeddingsClient, EmbeddingsConfig};
