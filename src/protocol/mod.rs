//! Protocol conversion with an explicit neutral representation and loss reporting.
//! Network authentication and backend capability policy belong to the caller.
pub mod events;
pub mod model;
pub mod request;
pub mod responses;
pub use events::{Decoder, Encoder, Event, Frame};
pub use model::*;
pub use request::{decode_request, encode_request, translate_request};

/// Render a completed wire response. Identity paths preserve all native fields.
pub fn translate_response(
    source: Protocol,
    target: Protocol,
    body: &serde_json::Value,
) -> Result<serde_json::Value, String> {
    if source == target {
        return Ok(body.clone());
    }
    match (source, target) {
        (Protocol::Responses, _) => {
            let mut decoder = Decoder::default();
            match decoder
                .decode(serde_json::json!({"type":"response.completed","response":body}))?
                .event
            {
                Event::Finish(completion) => completion.render(target),
                _ => Err("Missing completion".into()),
            }
        }
        (Protocol::Messages, Protocol::ChatCompletions) => {
            let response =
                serde_json::from_value::<crate::wire::anthropic::MessagesResponse>(body.clone())
                    .map_err(|e| e.to_string())?;
            serde_json::to_value(crate::convert::to_openai::anthropic_response_to_openai(
                response,
            ))
            .map_err(|e| e.to_string())
        }
        (Protocol::ChatCompletions, Protocol::Messages) => {
            let response =
                serde_json::from_value::<crate::wire::openai::ChatResponse>(body.clone())
                    .map_err(|e| e.to_string())?;
            let response = crate::convert::to_openai::response_to_anthropic(response)?;
            serde_json::to_value(response).map_err(|e| e.to_string())
        }
        _ => Err("Unsupported response conversion".into()),
    }
}

#[cfg(test)]
mod tests;
