//! Charge exactly what the provider reported.
//!
//! Every API returns a `usage` block with the real token counts, but each
//! provider counts prompt-cache tokens differently, and getting that wrong
//! either double-charges or under-charges:
//!
//! | API | Prompt count | Cache reads | Cache writes |
//! |---|---|---|---|
//! | OpenAI Chat Completions (`prompt_tokens`) | includes cached | `prompt_tokens_details.cached_tokens` (a subset) | not reported |
//! | OpenAI Responses (`input_tokens` + `input_tokens_details`) | includes cached | `input_tokens_details.cached_tokens` (a subset) | not reported |
//! | Anthropic Messages (`input_tokens`) | excludes cache | `cache_read_input_tokens` (extra) | `cache_creation_input_tokens` (extra) |
//! | Google Gemini (`usageMetadata.promptTokenCount`) | includes cached | `cachedContentTokenCount` (a subset) | not reported |
//!
//! [`Usage::from_json`] reads any of these (the whole response or just the
//! usage object) into one normalised [`Usage`], and
//! [`ModelPrice::cost_usage`](crate::pricing::ModelPrice::cost_usage) prices
//! it with the cache discounts.
//!
//! ```
//! use llm_budget::usage::Usage;
//!
//! // An Anthropic response: 100 fresh input tokens, 2,000 read from cache.
//! let body = r#"{"usage": {"input_tokens": 100, "output_tokens": 50,
//!                "cache_read_input_tokens": 2000, "cache_creation_input_tokens": 0}}"#;
//! let u = Usage::from_json(body)?;
//! assert_eq!(u.input_tokens, 2_100);          // all prompt tokens
//! assert_eq!(u.cached_input_tokens, 2_000);   // of which cheap cache reads
//! # Ok::<(), llm_budget::BudgetError>(())
//! ```

use crate::error::BudgetError;
use serde_json::Value;

/// Token counts of one call, normalised across providers.
///
/// `input_tokens` is every prompt token, including the ones served from or
/// written to the cache; `cached_input_tokens` and `cache_write_tokens` say how
/// many of them get the cache prices instead of the normal input price.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
pub struct Usage {
    /// All prompt tokens, cached or not.
    pub input_tokens: u64,
    /// Prompt tokens read from the provider's cache (part of `input_tokens`).
    pub cached_input_tokens: u64,
    /// Prompt tokens written to the provider's cache (part of `input_tokens`).
    pub cache_write_tokens: u64,
    /// Completion tokens, including any reasoning or "thinking" tokens.
    pub output_tokens: u64,
}

fn n(v: &Value, key: &str) -> u64 {
    v.get(key).and_then(Value::as_u64).unwrap_or(0)
}

impl Usage {
    /// Plain counts with no caching.
    pub fn new(input_tokens: u64, output_tokens: u64) -> Self {
        Self { input_tokens, output_tokens, ..Self::default() }
    }

    /// Prompt tokens billed at the normal input price.
    pub fn fresh_input_tokens(&self) -> u64 {
        self.input_tokens
            .saturating_sub(self.cached_input_tokens)
            .saturating_sub(self.cache_write_tokens)
    }

    /// Parse a provider response body, or just its usage object, from JSON text.
    ///
    /// # Errors
    /// [`BudgetError::InvalidConfig`] if the text is not JSON or has no
    /// recognisable usage block.
    pub fn from_json(text: &str) -> Result<Self, BudgetError> {
        let v: Value = serde_json::from_str(text)
            .map_err(|e| BudgetError::InvalidConfig(format!("usage JSON: {e}")))?;
        Self::from_value(&v)
    }

    /// Same as [`from_json`](Self::from_json) for an already parsed value.
    ///
    /// # Errors
    /// [`BudgetError::InvalidConfig`] if there is no recognisable usage block.
    pub fn from_value(v: &Value) -> Result<Self, BudgetError> {
        let u = v
            .get("usage")
            .or_else(|| v.get("usageMetadata"))
            .or_else(|| v.get("usage_metadata"))
            .unwrap_or(v);
        // OpenAI Chat Completions
        if u.get("prompt_tokens").is_some() {
            let cached = u.get("prompt_tokens_details").map_or(0, |d| n(d, "cached_tokens"));
            let input = n(u, "prompt_tokens");
            return Ok(Self {
                input_tokens: input,
                cached_input_tokens: cached.min(input),
                cache_write_tokens: 0,
                output_tokens: n(u, "completion_tokens"),
            });
        }
        // Google Gemini
        if u.get("promptTokenCount").is_some() || u.get("candidatesTokenCount").is_some() {
            let input = n(u, "promptTokenCount");
            return Ok(Self {
                input_tokens: input,
                cached_input_tokens: n(u, "cachedContentTokenCount").min(input),
                cache_write_tokens: 0,
                // thinking tokens are billed as output
                output_tokens: n(u, "candidatesTokenCount").saturating_add(n(u, "thoughtsTokenCount")),
            });
        }
        if u.get("input_tokens").is_some() || u.get("output_tokens").is_some() {
            // OpenAI Responses API: input includes the cached subset.
            if let Some(details) = u.get("input_tokens_details") {
                let input = n(u, "input_tokens");
                return Ok(Self {
                    input_tokens: input,
                    cached_input_tokens: n(details, "cached_tokens").min(input),
                    cache_write_tokens: 0,
                    output_tokens: n(u, "output_tokens"),
                });
            }
            // Anthropic: cache reads and writes come on top of input_tokens.
            let read = n(u, "cache_read_input_tokens");
            let write = n(u, "cache_creation_input_tokens");
            return Ok(Self {
                input_tokens: n(u, "input_tokens").saturating_add(read).saturating_add(write),
                cached_input_tokens: read,
                cache_write_tokens: write,
                output_tokens: n(u, "output_tokens"),
            });
        }
        Err(BudgetError::InvalidConfig(
            "no usage block found (expected OpenAI, Anthropic or Gemini usage fields)".into(),
        ))
    }
}

#[cfg(feature = "async-openai")]
impl From<&async_openai::types::chat::CompletionUsage> for Usage {
    fn from(u: &async_openai::types::chat::CompletionUsage) -> Self {
        let input = u64::from(u.prompt_tokens);
        let cached = u
            .prompt_tokens_details
            .as_ref()
            .and_then(|d| d.cached_tokens)
            .map_or(0, u64::from);
        Self {
            input_tokens: input,
            cached_input_tokens: cached.min(input),
            cache_write_tokens: 0,
            output_tokens: u64::from(u.completion_tokens),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pricing::PriceTable;

    #[test]
    fn openai_chat_cached_tokens_are_a_subset() {
        let u = Usage::from_json(
            r#"{"id":"x","usage":{"prompt_tokens":2006,"completion_tokens":300,"total_tokens":2306,
                "prompt_tokens_details":{"cached_tokens":1920},"completion_tokens_details":{"reasoning_tokens":0}}}"#,
        )
        .unwrap();
        assert_eq!(u, Usage { input_tokens: 2006, cached_input_tokens: 1920, cache_write_tokens: 0, output_tokens: 300 });
        assert_eq!(u.fresh_input_tokens(), 86);
    }

    #[test]
    fn openai_responses_api_is_not_double_counted() {
        // Same field names as Anthropic, opposite meaning: here input includes cache.
        let u = Usage::from_json(
            r#"{"usage":{"input_tokens":1000,"input_tokens_details":{"cached_tokens":800},
                "output_tokens":10,"output_tokens_details":{"reasoning_tokens":4},"total_tokens":1010}}"#,
        )
        .unwrap();
        assert_eq!(u.input_tokens, 1000);
        assert_eq!(u.cached_input_tokens, 800);
        assert_eq!(u.fresh_input_tokens(), 200);
    }

    #[test]
    fn anthropic_cache_counts_are_added() {
        let u = Usage::from_json(
            r#"{"input_tokens":50,"output_tokens":20,"cache_creation_input_tokens":1000,"cache_read_input_tokens":3000}"#,
        )
        .unwrap();
        assert_eq!(u.input_tokens, 4050);
        assert_eq!(u.fresh_input_tokens(), 50);
        assert_eq!(u.cache_write_tokens, 1000);
    }

    #[test]
    fn gemini_counts_thinking_as_output() {
        let u = Usage::from_json(
            r#"{"usageMetadata":{"promptTokenCount":500,"candidatesTokenCount":40,"thoughtsTokenCount":60,
                "cachedContentTokenCount":100,"totalTokenCount":600}}"#,
        )
        .unwrap();
        assert_eq!(u, Usage { input_tokens: 500, cached_input_tokens: 100, cache_write_tokens: 0, output_tokens: 100 });
    }

    #[test]
    fn bad_input_is_an_error_not_a_panic() {
        assert!(Usage::from_json("not json").is_err());
        assert!(Usage::from_json(r#"{"choices":[]}"#).is_err());
        // a cached count larger than the prompt is clamped
        let u = Usage::from_json(r#"{"prompt_tokens":10,"completion_tokens":1,"prompt_tokens_details":{"cached_tokens":99}}"#).unwrap();
        assert_eq!(u.fresh_input_tokens(), 0);
        let u = Usage::from_json(r#"{"prompt_tokens":-5,"completion_tokens":"x"}"#).unwrap();
        assert_eq!(u, Usage::default());
    }

    #[test]
    fn priced_with_cache_discounts() {
        // claude-sonnet-4-5: $3 in, $15 out, $0.30 cache read, $3.75 cache write per million.
        let p = PriceTable::builtin().require("claude-sonnet-4-5").unwrap();
        let u = Usage { input_tokens: 1_000_000 + 2_000_000 + 1_000_000, cached_input_tokens: 2_000_000, cache_write_tokens: 1_000_000, output_tokens: 1_000_000 };
        let expected = 3.0 + 2.0 * 0.3 + 3.75 + 15.0;
        assert!((p.cost_usage(&u) - expected).abs() < 1e-9, "{}", p.cost_usage(&u));
        // without caching it is the plain price
        assert!((p.cost_usage(&Usage::new(1_000, 1_000)) - p.cost(1_000, 1_000)).abs() < 1e-15);
    }

    #[cfg(feature = "async-openai")]
    #[test]
    fn converts_async_openai_usage() {
        let json = r#"{"prompt_tokens":2006,"completion_tokens":300,"total_tokens":2306,"prompt_tokens_details":{"cached_tokens":1920}}"#;
        let ao: async_openai::types::chat::CompletionUsage = serde_json::from_str(json).unwrap();
        assert_eq!(Usage::from(&ao), Usage::from_json(json).unwrap());
    }
}
