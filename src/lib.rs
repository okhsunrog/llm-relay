pub mod convert;
pub mod types;

#[cfg(feature = "client")]
pub mod client;

// Re-export commonly used types at crate root
pub use types::anthropic::{ContentBlock, Message, MessagesResponse};
pub use types::common::{
    EffortLevel, Provider, ResponseFormat, StopReason, ThinkingConfig, ToolDefinition, Usage,
};

#[cfg(feature = "client")]
pub use client::{AuthScheme, ChatOptions, ClientConfig, LlmClient, LlmError, RetryPolicy};

#[cfg(feature = "streaming")]
pub use client::{ChatStream, StreamEvent};

#[cfg(feature = "rig")]
pub use client::rig::RigClient;

#[cfg(feature = "embeddings")]
pub use client::{EmbeddingsClient, EmbeddingsConfig};
