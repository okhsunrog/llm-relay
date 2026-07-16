# llm-relay

Provider-neutral Rust types, protocol conversion, and HTTP transport for Anthropic and OpenAI-compatible LLM APIs. Anthropic-style content blocks are the canonical representation; provider-specific wire formats stay at the boundary.

## Why use it with Rig?

The crates solve different problems:

- **llm-relay** owns portable provider configuration, custom API URLs, authentication, custom headers, retries, response limits, canonical message conversion, embeddings, and normalized SSE events.
- **Rig** owns higher-level agent orchestration: tool loops, extractors, hooks, memory, and retrieval integrations.

Enable the `rig` feature to construct native Rig clients from the same `ClientConfig`. This keeps application settings independent of a specific provider without reimplementing Rig's agent runtime.

## Features

- Anthropic and OpenAI-compatible chat APIs with arbitrary HTTP(S) API bases
- Bidirectional message, tool-call, thinking, response, and usage conversion
- Normalized OpenAI/Anthropic SSE streaming, including tool argument deltas
- OpenAI-compatible embeddings with dimensions, input type, and encoding format
- Bearer, custom API-key-header, or no-auth operation
- Custom headers, configurable timeouts, bounded responses, and retry policy
- Optional Rig client adapters
- Types-only mode for proxies and protocol gateways

## Cargo features

| Feature | Default | Description |
|---|---:|---|
| `client` | yes | HTTP chat client |
| `embeddings` | no | OpenAI-compatible embeddings |
| `streaming` | no | Normalized SSE chat streams |
| `rig` | no | Build Rig OpenAI/Anthropic clients from `ClientConfig` |

```toml
llm-relay = { version = "0.3", features = ["embeddings", "streaming", "rig"] }
```

Use `default-features = false` for types and conversion without an HTTP runtime.

## API base URL contract

For OpenAI-compatible servers, `base_url` is the complete API base before the endpoint name. It commonly ends in `/v1`:

```text
http://localhost:11434/v1 + chat/completions
https://openrouter.ai/api/v1 + embeddings
```

For Anthropic-compatible servers, both the server root and a pasted `/v1` or `/v1/messages` URL are accepted and normalized.

## Chat

```rust,no_run
use llm_relay::{ChatOptions, ClientConfig, LlmClient, Message};

# async fn example() -> Result<(), Box<dyn std::error::Error>> {
let config = ClientConfig::openai_compatible(
    "https://llm.example.com/v1",
    "secret",
    "my-model",
)
.header("X-Tenant", "notes-rs");

let client = LlmClient::new(config)?;
let response = client
    .complete("Explain hybrid search", ChatOptions::default())
    .await?;
println!("{}", response.text());
# Ok(())
# }
```

Anthropic-compatible custom server:

```rust,no_run
# use llm_relay::{ClientConfig, LlmClient};
# fn example() -> Result<(), Box<dyn std::error::Error>> {
let client = LlmClient::new(
    ClientConfig::anthropic("secret", "claude-compatible-model")
        .base_url("https://anthropic-proxy.example.com"),
)?;
# Ok(())
# }
```

Local server without authentication:

```rust,no_run
# use llm_relay::{ClientConfig, LlmClient};
# fn example() -> Result<(), Box<dyn std::error::Error>> {
let client = LlmClient::new(ClientConfig::local_openai_compatible(
    "http://localhost:11434/v1",
    "qwen3",
))?;
# Ok(())
# }
```

## Structured output

OpenAI-compatible providers that implement `response_format.json_schema` can
return a value validated against a schema generated from the Rust type:

```rust,no_run
use llm_relay::{ClientConfig, LlmClient};
use schemars::JsonSchema;
use serde::Deserialize;

#[derive(Deserialize, JsonSchema)]
struct Keywords {
    values: Vec<String>,
}

# async fn example() -> Result<(), Box<dyn std::error::Error>> {
let client = LlmClient::new(ClientConfig::openrouter("secret", "google/gemini-3.1-flash-lite"))?;
let response = client
    .complete_structured::<Keywords>("Extract keywords from: Rust and SQLite", "keywords", None)
    .await?;
println!("{:?}", response.data.values);
# Ok(())
# }
```

The native Anthropic Messages transport rejects this method because its wire
format does not use OpenAI's `response_format` contract.

## Streaming

```rust,no_run
use futures_util::StreamExt;
use llm_relay::{ChatOptions, ClientConfig, LlmClient, Message, StreamEvent};

# async fn example() -> Result<(), Box<dyn std::error::Error>> {
let client = LlmClient::new(ClientConfig::openrouter("secret", "openai/gpt-5.4-mini"))?;
let mut stream = client
    .chat_stream(&[Message::user_text("Hello")], ChatOptions::default())
    .await?;

while let Some(event) = stream.next().await {
    match event? {
        StreamEvent::TextDelta { text } => print!("{text}"),
        StreamEvent::Usage { usage } => eprintln!("{} tokens", usage.total_tokens()),
        _ => {}
    }
}
# Ok(())
# }
```

## Embeddings

```rust,no_run
use llm_relay::{EmbeddingsClient, EmbeddingsConfig};

# async fn example() -> Result<(), Box<dyn std::error::Error>> {
let client = EmbeddingsClient::new(
    EmbeddingsConfig::openai_compatible(
        "https://embeddings.example.com/v1",
        "secret",
        "embedding-model",
    )
    .dimensions(1024)
    .input_type("document"),
)?;
let vectors = client.create_embeddings(&["first", "second"]).await?;
# Ok(())
# }
```

## Rig adapter

```rust,no_run
use llm_relay::{ClientConfig, RigClient};
use rig::client::CompletionClient;

# fn example() -> Result<(), Box<dyn std::error::Error>> {
let config = ClientConfig::openai_compatible(
    "http://localhost:8000/v1",
    "",
    "local-model",
);

match config.rig_client()? {
    RigClient::OpenAi(client) => {
        let _agent = client.agent(&config.model).build();
    }
    RigClient::Anthropic(client) => {
        let _agent = client.agent(&config.model).build();
    }
}
# Ok(())
# }
```

Not every “compatible” server implements streaming, tool calling, structured output, or usage reporting. Applications should probe the capabilities they actually need instead of assuming complete compatibility.
