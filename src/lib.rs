mod convert;
pub mod wire;

#[cfg(feature = "client")]
pub mod client;

// Re-export commonly used types at crate root
pub use protocol::{Content, Item, Opaque, Policy, Protocol, Request};
/// Provider-specific request preparation utilities, outside protocol conversion.
pub mod anthropic {
    pub use crate::convert::{cache_control, thinking, tool_names};
}
pub use wire::common::{
    EffortLevel, Provider, ResponseFormat, StopReason, ThinkingConfig, ToolDefinition, Usage,
};

#[cfg(feature = "client")]
pub use client::{
    AuthScheme, ClientConfig, LlmError, RetryPolicy, StructuredResponse, WireChatOptions,
    WireClient,
};

#[cfg(feature = "streaming")]
pub use client::{WireChatStream, WireStreamEvent};

#[cfg(feature = "rig")]
pub use client::rig::RigClient;

#[cfg(feature = "embeddings")]
pub use client::{EmbeddingsClient, EmbeddingsConfig};

/// Loss-aware protocol adapters independent of HTTP/authentication.
pub mod protocol;
