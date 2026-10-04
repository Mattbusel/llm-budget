//! Real token counts from text, using OpenAI's tokenizers via
//! [`tiktoken-rs`](https://crates.io/crates/tiktoken-rs). Needs the `tokens` feature.
//!
//! For OpenAI models the count is exact (the same BPE the API bills with).
//! Other vendors (Anthropic, Google, Mistral, ...) do not publish their
//! tokenizers, so for them the count uses OpenAI's `o200k_base` encoding and is
//! marked `exact: false`. Treat it as an estimate and, where you can, charge
//! the `usage` numbers the API returns instead.
//!
//! ```
//! use llm_budget::tokens::count_tokens;
//!
//! let n = count_tokens("gpt-4o", "Hello, world!");
//! assert_eq!(n.tokens, 4);
//! assert!(n.exact);
//! ```

use crate::error::BudgetError;
use crate::pricing::PriceTable;
use tiktoken_rs::tokenizer::{get_tokenizer, Tokenizer};

/// A token count and whether it came from the model's own tokenizer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TokenCount {
    /// Number of tokens.
    pub tokens: usize,
    /// True when the model's own tokenizer was used (OpenAI models), false
    /// when `o200k_base` stood in for an unknown tokenizer.
    pub exact: bool,
}

/// Count the tokens in `text` as `model` would see them.
pub fn count_tokens(model: &str, text: &str) -> TokenCount {
    let bare = model.rsplit_once('/').map_or(model, |(_, m)| m);
    let (tokenizer, exact) = match get_tokenizer(bare) {
        Some(t) => (t, true),
        None => (Tokenizer::O200kBase, false),
    };
    let bpe = tiktoken_rs::bpe_for_tokenizer(tokenizer)
        .unwrap_or_else(|_| tiktoken_rs::o200k_base_singleton());
    TokenCount { tokens: bpe.encode_ordinary(text).len(), exact }
}

/// Price of sending `prompt` to `model` and getting `completion` back, using
/// [`count_tokens`] and the built-in [`PriceTable`].
///
/// # Errors
/// [`BudgetError::UnknownModel`] if the model has no price.
pub fn cost_of_text(model: &str, prompt: &str, completion: &str) -> Result<f64, BudgetError> {
    let price = PriceTable::builtin().require(model)?;
    let input = count_tokens(model, prompt).tokens as u64;
    let output = count_tokens(model, completion).tokens as u64;
    Ok(price.cost(input, output))
}

impl crate::budget::AgentBudget {
    /// Count the tokens of `prompt` and `completion` with the model's tokenizer
    /// and charge them at the model's price. Needs the `tokens` feature.
    ///
    /// # Errors
    /// [`BudgetError::UnknownModel`], [`BudgetError::InvalidConfig`] if a text
    /// has more than `u32::MAX` tokens, or any error from a normal charge.
    pub async fn charge_model_text(
        &self,
        model: &str,
        prompt: &str,
        completion: &str,
        task_id: Option<String>,
        session_id: Option<String>,
    ) -> Result<crate::budget::ChargeReceipt, BudgetError> {
        let to_u32 = |n: usize| {
            u32::try_from(n).map_err(|_| BudgetError::InvalidConfig("text too long to count".into()))
        };
        let input = to_u32(count_tokens(model, prompt).tokens)?;
        let output = to_u32(count_tokens(model, completion).tokens)?;
        self.charge_model(model, input, output, task_id, session_id).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn openai_models_are_exact() {
        // Reference counts from OpenAI's tiktoken (o200k_base and cl100k_base).
        assert_eq!(count_tokens("gpt-4o", "Hello, world!"), TokenCount { tokens: 4, exact: true });
        assert_eq!(count_tokens("gpt-4", "Hello, world!"), TokenCount { tokens: 4, exact: true });
        assert_eq!(count_tokens("gpt-4o", ""), TokenCount { tokens: 0, exact: true });
        assert!(count_tokens("openai/gpt-4o-mini", "hi").exact);
    }

    #[test]
    fn other_vendors_are_marked_estimates() {
        let n = count_tokens("claude-sonnet-4-5", "Hello, world!");
        assert!(!n.exact);
        assert_eq!(n.tokens, 4);
    }

    #[test]
    fn multibyte_text_is_counted_not_byte_split() {
        // 4 bytes per char: a bytes/4 guess would say 10, the tokenizer knows better.
        let text = "\u{1F600}".repeat(10);
        let n = count_tokens("gpt-4o", &text).tokens;
        assert!((1..=20).contains(&n), "{n}");
    }

    #[test]
    fn cost_of_text_uses_counts_and_price() {
        let usd = cost_of_text("gpt-4o", "Hello, world!", "Hi").expect("price");
        let expected = (4.0 * 2.5 + count_tokens("gpt-4o", "Hi").tokens as f64 * 10.0) / 1_000_000.0;
        assert!((usd - expected).abs() < 1e-15);
        assert!(cost_of_text("no-such-model", "a", "b").is_err());
    }

    #[tokio::test]
    async fn charge_model_text_charges_counted_tokens() {
        let agent = crate::AgentBudget::new("a", 1.0, crate::ModelTier::Standard, 0.1).expect("new");
        let r = agent
            .charge_model_text("gpt-4o", "Hello, world!", "", None, None)
            .await
            .expect("charge");
        assert!((r.cost_usd - 0.00001).abs() < 1e-12, "{}", r.cost_usd);
        let log = agent.audit_log().await;
        assert_eq!(log[0].input_tokens, 4);
    }
}
