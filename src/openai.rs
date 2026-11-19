//! Dollar caps for [async-openai](https://docs.rs/async-openai) chat
//! completions (feature `async-openai`).
//!
//! The pattern is the crate's core one, reserve then settle, with the
//! async-openai types doing the bookkeeping:
//!
//! 1. [`reserve_request`] holds the estimated cost of a
//!    `CreateChatCompletionRequest` (prompt size from its messages, output
//!    from `max_completion_tokens`) and refuses when it does not fit, so the
//!    call is never sent.
//! 2. After the call, [`settle_response`] books what the response's `usage`
//!    says it cost. For a stream, feed every chunk to a [`StreamUsage`] and
//!    settle with it at the end: the last chunk carries the usage when the
//!    request set `stream_options.include_usage` ([`include_usage`] does that).
//!
//! The crate only depends on async-openai's types, so this works with any
//! client that speaks them, including OpenAI-compatible servers.
//!
//! ```
//! use async_openai::types::chat::{CreateChatCompletionRequest, CreateChatCompletionResponse};
//! use llm_budget::{openai, AgentBudget};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let budget = AgentBudget::builder("support-bot", 1.00).build()?;
//! let request: CreateChatCompletionRequest = serde_json::from_str(
//!     r#"{"model":"gpt-4o","max_completion_tokens":500,
//!         "messages":[{"role":"user","content":"Where is my order?"}]}"#,
//! )?;
//!
//! let hold = openai::reserve_request(&budget, &request)?; // refused here if over the cap
//! // let response = client.chat().create(request).await?;
//! # let response: CreateChatCompletionResponse = serde_json::from_str(
//! #     r#"{"id":"c1","object":"chat.completion","created":0,"model":"gpt-4o-2024-08-06","choices":[],
//! #         "usage":{"prompt_tokens":1200,"completion_tokens":80,"total_tokens":1280}}"#)?;
//! let outcome = openai::settle_response(hold, &response);
//! assert!(!outcome.over_cap);
//! assert_eq!(budget.reserved_usd(), 0.0);
//! # Ok(())
//! # }
//! ```

use crate::budget::{AgentBudget, ChargeContext, ChargeOutcome, Reservation};
use crate::error::BudgetError;
use crate::usage::{estimate_json_tokens, Usage};
use async_openai::types::chat::{
    ChatCompletionStreamOptions, CreateChatCompletionRequest, CreateChatCompletionResponse,
    CreateChatCompletionStreamResponse,
};

/// Expected size of a request: prompt tokens estimated from its messages and
/// tools (about four characters per token), and output tokens from
/// `max_completion_tokens` (or the older `max_tokens`) times `n`. Zero output
/// tokens when neither is set, so set one to reserve for the answer too.
pub fn estimate_request(request: &CreateChatCompletionRequest) -> Usage {
    let messages = serde_json::to_value(&request.messages).map_or(0, |v| estimate_json_tokens(&v));
    let tools =
        request.tools.as_ref().and_then(|t| serde_json::to_value(t).ok()).map_or(0, |v| estimate_json_tokens(&v));
    // a few tokens of framing per message
    let framing = 4 * request.messages.len() as u64;
    #[allow(deprecated)]
    let max_out = request.max_completion_tokens.or(request.max_tokens).map_or(0, u64::from);
    let n = u64::from(request.n.unwrap_or(1).max(1));
    Usage::new(messages + tools + framing, max_out.saturating_mul(n))
}

/// Hold the estimated cost of `request` (see [`estimate_request`]) on
/// `budget`, priced for `request.model` from the budget's price table.
///
/// # Errors
/// [`BudgetError::UnknownModel`] if the model has no price, otherwise the
/// errors of [`AgentBudget::reserve`] (exhausted, killed).
pub fn reserve_request(
    budget: &AgentBudget,
    request: &CreateChatCompletionRequest,
) -> Result<Reservation, BudgetError> {
    budget.reserve_for(&request.model, &estimate_request(request))
}

/// Settle `reservation` with the usage of a chat completion response.
///
/// Priced for `response.model`, falling back to the model the hold was
/// reserved for (the response usually echoes a dated id). A response without
/// `usage` books the reserved estimate. Never fails; books past the cap and
/// sets [`ChargeOutcome::over_cap`].
pub fn settle_response(reservation: Reservation, response: &CreateChatCompletionResponse) -> ChargeOutcome {
    settle(reservation, &response.model, response.usage.as_ref().map(Usage::from))
}

/// Book a finished chat completion that had no reservation (see
/// [`AgentBudget::record`]).
///
/// # Errors
/// [`BudgetError::InvalidConfig`] if the response has no `usage`, and
/// [`BudgetError::UnknownModel`] if its model has no price. Nothing is booked
/// on error.
pub fn record_response(
    budget: &AgentBudget,
    response: &CreateChatCompletionResponse,
) -> Result<ChargeOutcome, BudgetError> {
    let usage = response
        .usage
        .as_ref()
        .ok_or_else(|| BudgetError::InvalidConfig("chat completion response has no usage".into()))?;
    budget.record(&response.model, &Usage::from(usage))
}

/// Ask for the usage chunk at the end of a stream
/// (`stream_options.include_usage = true`), keeping any other stream options.
/// Without it the stream carries no token counts and only the estimate can
/// be booked.
pub fn include_usage(request: &mut CreateChatCompletionRequest) {
    let opts = request.stream_options.get_or_insert_with(ChatCompletionStreamOptions::default);
    opts.include_usage = Some(true);
}

/// Collects the model and the final usage of a streamed chat completion.
///
/// Call [`observe`](Self::observe) on every chunk, then
/// [`settle`](Self::settle) once the stream ends (or is dropped early: the
/// reserved estimate is booked when no usage chunk arrived, since the
/// provider bills the tokens it produced).
///
/// ```
/// use async_openai::types::chat::CreateChatCompletionStreamResponse;
/// use llm_budget::{openai::StreamUsage, AgentBudget, Usage};
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let budget = AgentBudget::builder("chat", 1.00).build()?;
/// let hold = budget.reserve_for("gpt-4o-mini", &Usage::new(2_000, 1_000))?;
/// let mut collected = StreamUsage::new();
/// for line in [
///     r#"{"id":"c","object":"chat.completion.chunk","created":0,"model":"gpt-4o-mini","choices":[],"usage":null}"#,
///     r#"{"id":"c","object":"chat.completion.chunk","created":0,"model":"gpt-4o-mini","choices":[],
///         "usage":{"prompt_tokens":1800,"completion_tokens":420,"total_tokens":2220}}"#,
/// ] {
///     let chunk: CreateChatCompletionStreamResponse = serde_json::from_str(line)?;
///     collected.observe(&chunk);
/// }
/// assert_eq!(collected.usage(), Some(Usage::new(1_800, 420)));
/// let outcome = collected.settle(hold);
/// assert!((outcome.recorded_usd - (1_800.0 * 0.15 + 420.0 * 0.6) / 1e6).abs() < 1e-12);
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone, Default)]
pub struct StreamUsage {
    model: Option<String>,
    usage: Option<Usage>,
}

impl StreamUsage {
    /// Nothing seen yet.
    pub fn new() -> Self {
        Self::default()
    }

    /// Note the model and, on the final chunk, the usage of one stream chunk.
    pub fn observe(&mut self, chunk: &CreateChatCompletionStreamResponse) {
        if self.model.is_none() && !chunk.model.is_empty() {
            self.model = Some(chunk.model.clone());
        }
        if let Some(u) = &chunk.usage {
            self.usage = Some(Usage::from(u));
        }
    }

    /// The usage from the final chunk, once it has arrived.
    pub fn usage(&self) -> Option<Usage> {
        self.usage
    }

    /// The model the chunks named.
    pub fn model(&self) -> Option<&str> {
        self.model.as_deref()
    }

    /// Settle `reservation` with what the stream reported: the usage priced
    /// for the streamed model (falling back to the reserved one), or the
    /// reserved estimate when no usage chunk arrived. Never fails.
    pub fn settle(self, reservation: Reservation) -> ChargeOutcome {
        let model = self.model.or_else(|| reservation.model().map(str::to_string)).unwrap_or_default();
        settle(reservation, &model, self.usage)
    }
}

fn settle(reservation: Reservation, model: &str, usage: Option<Usage>) -> ChargeOutcome {
    match usage {
        Some(u) => reservation.settle_usage(model, &u),
        None => {
            let estimate = reservation.estimate_usd();
            let model = reservation.model().unwrap_or(model).to_string();
            reservation.settle_with(estimate, ChargeContext::new().model(model))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PriceTable;

    fn request(json: &str) -> CreateChatCompletionRequest {
        serde_json::from_str(json).unwrap()
    }

    #[test]
    fn estimates_prompt_and_output() {
        let r = request(
            r#"{"model":"gpt-4o","max_completion_tokens":100,"n":2,
                "messages":[{"role":"system","content":"abcdefgh"},{"role":"user","content":"abcd"}]}"#,
        );
        let u = estimate_request(&r);
        // "system" (2) + "abcdefgh" (2) + "user" (1) + "abcd" (1) + 2 x 4 framing
        assert_eq!(u.input_tokens, 14);
        assert_eq!(u.output_tokens, 200);
        let legacy = request(r#"{"model":"gpt-4o","max_tokens":7,"messages":[]}"#);
        assert_eq!(estimate_request(&legacy), Usage::new(0, 7));
    }

    #[test]
    fn reserve_then_settle_books_reported_usage() {
        let budget = AgentBudget::builder("a", 1.0).build().unwrap();
        let r =
            request(r#"{"model":"gpt-4o","max_completion_tokens":1000,"messages":[{"role":"user","content":"hi"}]}"#);
        let hold = reserve_request(&budget, &r).unwrap();
        assert!(budget.reserved_usd() > 0.0);
        let resp: CreateChatCompletionResponse = serde_json::from_str(
            r#"{"id":"x","object":"chat.completion","created":1,"model":"acme-router-v1","choices":[],
                "usage":{"prompt_tokens":2006,"completion_tokens":300,"total_tokens":2306,
                         "prompt_tokens_details":{"cached_tokens":1920}}}"#,
        )
        .unwrap();
        let out = settle_response(hold, &resp);
        // the echoed id has no price, so the reserved model (gpt-4o) prices it
        // (a dated snapshot such as gpt-4o-2099-01-01 now resolves through
        // llm-prices 1.3 and is priced as itself)
        let expected =
            PriceTable::builtin().cost_usage("gpt-4o", &Usage::new(2006, 300).with_cache_read(1920)).unwrap();
        assert!((out.recorded_usd - expected).abs() < 1e-9);
        assert_eq!(budget.reserved_usd(), 0.0);
        let log = budget.drain_audit_log();
        assert_eq!(log[0].model.as_deref(), Some("gpt-4o"));
        assert_eq!(log[0].input_tokens, 2006);
    }

    #[test]
    fn missing_usage_books_the_estimate() {
        let budget = AgentBudget::builder("a", 1.0).build().unwrap();
        let r = request(r#"{"model":"gpt-4o","max_completion_tokens":1000,"messages":[]}"#);
        let hold = reserve_request(&budget, &r).unwrap();
        let held = hold.estimate_usd();
        let resp: CreateChatCompletionResponse =
            serde_json::from_str(r#"{"id":"x","object":"chat.completion","created":1,"model":"gpt-4o","choices":[]}"#)
                .unwrap();
        assert!(record_response(&budget, &resp).is_err());
        let out = settle_response(hold, &resp);
        assert!((out.recorded_usd - held).abs() < 1e-12);
        assert!((budget.spent_usd() - held).abs() < 1e-12);
    }

    #[test]
    fn refused_when_over_cap() {
        let budget = AgentBudget::builder("a", 0.001).build().unwrap();
        let r = request(r#"{"model":"gpt-4o","max_completion_tokens":100000,"messages":[]}"#);
        assert!(matches!(reserve_request(&budget, &r), Err(BudgetError::AgentExhausted { .. })));
        let unknown = request(r#"{"model":"not-a-model","messages":[]}"#);
        assert!(matches!(reserve_request(&budget, &unknown), Err(BudgetError::UnknownModel(_))));
    }

    #[test]
    fn record_without_reservation() {
        let budget = AgentBudget::builder("a", 1.0).build().unwrap();
        let resp: CreateChatCompletionResponse = serde_json::from_str(
            r#"{"id":"x","object":"chat.completion","created":1,"model":"gpt-4o-mini","choices":[],
                "usage":{"prompt_tokens":1000000,"completion_tokens":0,"total_tokens":1000000}}"#,
        )
        .unwrap();
        let out = record_response(&budget, &resp).unwrap();
        assert!((out.recorded_usd - 0.15).abs() < 1e-9);
    }

    #[test]
    fn stream_usage_from_final_chunk() {
        let budget = AgentBudget::builder("a", 1.0).build().unwrap();
        let mut r = request(r#"{"model":"gpt-4o-mini","stream":true,"messages":[]}"#);
        include_usage(&mut r);
        assert_eq!(r.stream_options.and_then(|o| o.include_usage), Some(true));

        let hold = budget.reserve_for("gpt-4o-mini", &Usage::new(10, 10)).unwrap();
        let mut s = StreamUsage::new();
        let chunk = |usage: &str| -> CreateChatCompletionStreamResponse {
            serde_json::from_str(&format!(
                r#"{{"id":"c","object":"chat.completion.chunk","created":0,"model":"gpt-4o-mini","choices":[],"usage":{usage}}}"#
            ))
            .unwrap()
        };
        s.observe(&chunk("null"));
        assert_eq!(s.usage(), None);
        assert_eq!(s.model(), Some("gpt-4o-mini"));
        s.observe(&chunk(r#"{"prompt_tokens":1000000,"completion_tokens":1000000,"total_tokens":2000000}"#));
        let out = s.settle(hold);
        assert!((out.recorded_usd - 0.75).abs() < 1e-9);
        assert!(out.over_cap == (budget.spent_usd() > 1.0));
    }

    #[test]
    fn interrupted_stream_books_the_estimate() {
        let budget = AgentBudget::builder("a", 1.0).build().unwrap();
        let hold = budget.reserve_for("gpt-4o-mini", &Usage::new(100_000, 10_000)).unwrap();
        let held = hold.estimate_usd();
        let out = StreamUsage::new().settle(hold);
        assert!((out.recorded_usd - held).abs() < 1e-12);
        assert_eq!(budget.drain_audit_log()[0].model.as_deref(), Some("gpt-4o-mini"));
    }
}
