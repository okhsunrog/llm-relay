# llm-relay

Shared Rust crate for LLM API types, format conversion, and HTTP client. Uses Anthropic as the canonical internal format and converts at the API boundary.

## Features

- **Anthropic & OpenAI types** — typed content blocks, tool use, extended thinking, embeddings
- **Bidirectional conversion** — Anthropic <-> OpenAI format (messages, tools, responses)
- **HTTP client** — chat completions (both providers) and embeddings
- **Extended thinking** — adaptive thinking (Opus 4.6, Sonnet 4.6) and manual budget mode
- **Proxy utilities** — cache control injection, MCP tool name transforms

## Cargo features

| Feature | Default | Description |
|---|---|---|
| `client` | yes | HTTP client (reqwest + thiserror) |
| `embeddings` | no | Embeddings client (implies `client`) |

Use `default-features = false` for types and conversion only (no HTTP dependencies).

## Usage

```rust
use llm_relay::{LlmClient, ClientConfig, ChatOptions};
use llm_relay::types::anthropic::Message;

let client = LlmClient::new(
    ClientConfig::anthropic("sk-...", "claude-sonnet-4-5-20250514")
)?;

// Simple text completion
let response = client.complete(Some("You are helpful."), "Hello!", None).await?;

// Full chat with tools and thinking
let messages = vec![Message::user_text("What's the weather?")];
let options = ChatOptions { system: Some("..."), tools: Some(&tools), thinking: Some(&thinking) };
let response = client.chat(&messages, options).await?;
```

Types-only (for proxy/conversion scenarios):

```rust
// Cargo.toml: llm-relay = { path = "...", default-features = false }
use llm_relay::convert::to_anthropic::inbound_request_to_anthropic;
use llm_relay::convert::to_openai::anthropic_response_to_openai;
use llm_relay::types::openai::InboundChatRequest;
```
