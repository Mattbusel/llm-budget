//! Dollar caps for [genai](https://docs.rs/genai) chat calls (feature `genai`).
//!
//! genai talks to OpenAI, Anthropic, Gemini, DeepSeek, Ollama, xAI, Groq and
//! more through one `Client`. These helpers wrap a call in the crate's core
//! pattern:
//!
//! 1. Reserve the estimated cost (prompt size from the [`ChatRequest`], output
//!    from `ChatOptions::max_tokens`) with
//!    [`AgentBudget::reserve_for`]. If it does not fit, the call is never sent
//!    and you get [`Error::Budget`].
//! 2. Make the call. If genai fails, the hold is released and nothing is
//!    booked.
//! 3. Settle with the usage genai reports, which it already normalises across
//!    providers (prompt tokens include cache reads and writes).
//!
//! ```no_run
//! use genai::chat::{ChatMessage, ChatOptions, ChatRequest};
//! use llm_budget::AgentBudget;
//!
//! # async fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let client = genai::Client::new()?;
//! let budget = AgentBudget::builder("researcher", 0.50).build()?;
//! let req = ChatRequest::new(vec![ChatMessage::user("Why is the sky blue?")]);
//! let opts = ChatOptions::default().with_max_tokens(500);
//!
//! let (res, outcome) = llm_budget::genai::exec_chat(&client, &budget, "gpt-4o-mini", req, Some(&opts)).await?;
//! println!("{} (${:.5})", res.first_text().unwrap_or(""), outcome.recorded_usd);
//! if outcome.over_cap {
//!     // stop the agent loop
//! }
//! # Ok(())
//! # }
//! ```
//!
//! For streams, [`exec_chat_stream`] turns on `capture_usage` and hands back
//! the stream with its [`Reservation`]; settle it with [`settle_stream_end`]
//! when the `End` event arrives.
//!
//! Model names are priced from the budget's
//! [`PriceTable`](crate::PriceTable); a genai namespace prefix such as
//! `openai::` is ignored for pricing.

use crate::budget::{AgentBudget, ChargeOutcome, Reservation};
use crate::error::BudgetError;
use crate::usage::{estimate_json_tokens, Usage};
use ::genai::chat::{ChatOptions, ChatRequest, ChatResponse, ChatStreamResponse, StreamEnd};
use ::genai::Client;

/// What can go wrong running a budgeted genai call.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// The budget refused the call (it was not sent), or the model has no price.
    #[error(transparent)]
    Budget(#[from] BudgetError),
    /// genai failed; the reservation was released and nothing was booked.
    #[error(transparent)]
    Genai(#[from] ::genai::Error),
}

fn n(v: Option<i32>) -> u64 {
    v.and_then(|x| u64::try_from(x).ok()).unwrap_or(0)
}

/// genai's normalised usage: `prompt_tokens` already includes cached and
/// cache-written tokens, and `completion_tokens` includes reasoning tokens.
impl From<&::genai::chat::Usage> for Usage {
    fn from(u: &::genai::chat::Usage) -> Self {
        let input = n(u.prompt_tokens);
        let details = u.prompt_tokens_details.as_ref();
        let cached = n(details.and_then(|d| d.cached_tokens)).min(input);
        let write = details.map_or(0, |d| {
            let by_ttl = d
                .cache_creation_details
                .as_ref()
                .map_or(0, |c| n(c.ephemeral_5m_tokens) + n(c.ephemeral_1h_tokens));
            n(d.cache_creation_tokens).max(by_ttl)
        });
        Usage {
            input_tokens: input,
            cached_input_tokens: cached,
            cache_write_tokens: write.min(input - cached),
            output_tokens: n(u.completion_tokens),
        }
    }
}

/// The part of a genai model spec used for pricing: `openai::gpt-4o` -> `gpt-4o`.
fn price_key(model: &str) -> &str {
    model.rsplit_once("::").map_or(model, |(_, m)| m)
}

/// Expected size of a call: prompt tokens estimated from the request's system
/// prompt, messages and tools (about four characters per token, plus a few
/// per message), output tokens from `options.max_tokens` (zero when unset, so
/// set it to reserve for the answer too).
pub fn estimate_request(request: &ChatRequest, options: Option<&ChatOptions>) -> Usage {
    let prompt = serde_json::to_value(request).map_or(0, |v| estimate_json_tokens(&v));
    let framing = 4 * (request.messages.len() as u64 + u64::from(request.system.is_some()));
    let output = options.and_then(|o| o.max_tokens).map_or(0, u64::from);
    Usage::new(prompt + framing, output)
}

/// Run `client.exec_chat` under a hold on `budget`, and book what it cost.
///
/// Returns genai's response and what was booked. The call is not sent when
/// the estimate does not fit ([`Error::Budget`]); a genai error releases the
/// hold ([`Error::Genai`]). A successful call is always booked, even past the
/// cap: check [`ChargeOutcome::over_cap`] before the next one.
///
/// # Errors
/// See above.
pub async fn exec_chat(
    client: &Client,
    budget: &AgentBudget,
    model: &str,
    request: ChatRequest,
    options: Option<&ChatOptions>,
) -> Result<(ChatResponse, ChargeOutcome), Error> {
    let hold = budget.reserve_for(price_key(model), &estimate_request(&request, options))?;
    let response = client.exec_chat(model, request, options).await?;
    let outcome = settle_response(hold, &response);
    Ok((response, outcome))
}

/// Settle `reservation` with the usage of a genai [`ChatResponse`], priced
/// for the model the provider reports (falling back to the reserved model,
/// then to the reserved estimate). Never fails.
pub fn settle_response(reservation: Reservation, response: &ChatResponse) -> ChargeOutcome {
    let model = response.provider_model_iden.model_name.to_string();
    reservation.settle_usage(price_key(&model), &Usage::from(&response.usage))
}

/// Start `client.exec_chat_stream` under a hold on `budget`, with
/// `capture_usage` turned on so the stream's `End` event carries the usage.
///
/// Returns the stream and its [`Reservation`]: pass the reservation and the
/// `End` event to [`settle_stream_end`]. If the stream is abandoned, call
/// [`Reservation::settle`] with the estimate (the provider bills what it
/// produced) or drop the reservation to book nothing.
///
/// # Errors
/// [`Error::Budget`] when the estimate does not fit (nothing is sent),
/// [`Error::Genai`] when the stream cannot be started (the hold is released).
pub async fn exec_chat_stream(
    client: &Client,
    budget: &AgentBudget,
    model: &str,
    request: ChatRequest,
    options: Option<&ChatOptions>,
) -> Result<(ChatStreamResponse, Reservation), Error> {
    let hold = budget.reserve_for(price_key(model), &estimate_request(&request, options))?;
    let options = options.cloned().unwrap_or_default().with_capture_usage(true);
    let stream = client.exec_chat_stream(model, request, Some(&options)).await?;
    Ok((stream, hold))
}

/// Settle a stream's reservation from its `End` event. Without captured usage
/// (the stream was started without `capture_usage`), the reserved estimate is
/// booked. Never fails.
pub fn settle_stream_end(reservation: Reservation, end: &StreamEnd) -> ChargeOutcome {
    match &end.captured_usage {
        Some(u) => {
            let model = reservation.model().unwrap_or_default().to_string();
            reservation.settle_usage(&model, &Usage::from(u))
        }
        None => {
            let estimate = reservation.estimate_usd();
            reservation.settle(estimate)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ::genai::chat::{CacheCreationDetails, ChatMessage, PromptTokensDetails};

    fn usage(prompt: i32, cached: Option<i32>, write: Option<i32>, out: i32) -> ::genai::chat::Usage {
        ::genai::chat::Usage {
            prompt_tokens: Some(prompt),
            prompt_tokens_details: Some(PromptTokensDetails {
                cache_creation_tokens: write,
                cache_creation_details: None,
                cached_tokens: cached,
                audio_tokens: None,
            }),
            completion_tokens: Some(out),
            completion_tokens_details: None,
            total_tokens: Some(prompt + out),
        }
    }

    #[test]
    fn converts_genai_usage() {
        // An Anthropic call as genai reports it: 300 fresh + 11,000 read + 500 written.
        let u = Usage::from(&usage(11_800, Some(11_000), Some(500), 250));
        assert_eq!(u, Usage::from_parts(300, 11_000, 500, 250));
        // TTL details count when the total is missing
        let mut g = usage(1_000, None, None, 1);
        if let Some(d) = g.prompt_tokens_details.as_mut() {
            d.cache_creation_details =
                Some(CacheCreationDetails { ephemeral_5m_tokens: Some(100), ephemeral_1h_tokens: Some(50) });
        }
        assert_eq!(Usage::from(&g).cache_write_tokens, 150);
        // nonsense is clamped, never panics
        assert_eq!(Usage::from(&usage(-5, Some(99), Some(99), -1)), Usage::default());
        assert_eq!(Usage::from(&::genai::chat::Usage::default()), Usage::default());
    }

    #[test]
    fn estimates_and_price_keys() {
        let req = ChatRequest::new(vec![ChatMessage::user("abcdefgh")]).with_system("abcd");
        let opts = ChatOptions::default().with_max_tokens(300);
        let u = estimate_request(&req, Some(&opts));
        assert!(u.input_tokens >= 3 + 8, "{u:?}");
        assert_eq!(u.output_tokens, 300);
        assert_eq!(estimate_request(&req, None).output_tokens, 0);
        assert_eq!(price_key("openai::gpt-4o"), "gpt-4o");
        assert_eq!(price_key("gpt-4o"), "gpt-4o");
    }

    #[test]
    fn stream_end_settles_with_captured_usage_or_the_estimate() {
        let budget = AgentBudget::builder("a", 1.0).build().unwrap();
        let end = |u: Option<::genai::chat::Usage>| StreamEnd {
            captured_usage: u,
            captured_stop_reason: None,
            captured_content: None,
            captured_reasoning_content: None,
            captured_response_id: None,
        };
        let hold = budget.reserve_for("gpt-4o-mini", &Usage::new(1_000, 1_000)).unwrap();
        let out = settle_stream_end(hold, &end(Some(usage(1_000_000, None, None, 0))));
        assert!((out.recorded_usd - 0.15).abs() < 1e-9);

        let hold = budget.reserve_for("gpt-4o-mini", &Usage::new(1_000, 1_000)).unwrap();
        let held = hold.estimate_usd();
        let out = settle_stream_end(hold, &end(None));
        assert!((out.recorded_usd - held).abs() < 1e-12);
        assert_eq!(budget.reserved_usd(), 0.0);
    }
}
