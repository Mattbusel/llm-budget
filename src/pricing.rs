//! Per-model prices, so a charge can use the real price of the model you called
//! instead of a rough tier average.
//!
//! The built-in table is generated from LiteLLM's maintained
//! `model_prices_and_context_window.json` (MIT license) by
//! `scripts/update_prices.py`. It covers the chat models of OpenAI, Anthropic,
//! Google Gemini, Mistral, DeepSeek and xAI. [`SNAPSHOT_DATE`] says when it was
//! generated. Prices change, so you can also load a newer copy of that JSON file
//! at runtime with [`PriceTable::from_litellm_json`].
//!
//! ```
//! use llm_budget::pricing::PriceTable;
//!
//! let table = PriceTable::builtin();
//! let price = table.get("gpt-4o").expect("gpt-4o is in the table");
//! // 1,000 input tokens and 500 output tokens:
//! let usd = price.cost(1_000, 500);
//! assert!((usd - (1_000.0 * 2.5 + 500.0 * 10.0) / 1_000_000.0).abs() < 1e-12);
//! ```

use crate::error::BudgetError;
use std::collections::HashMap;
use std::sync::OnceLock;

#[path = "prices_data.rs"]
mod data;

pub use data::SNAPSHOT_DATE;

/// Price of one model, in US dollars per million tokens.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
pub struct ModelPrice {
    /// Model name as the provider's API spells it, for example `gpt-4o`.
    pub model: String,
    /// Provider name as LiteLLM spells it, for example `openai` or `anthropic`.
    pub provider: String,
    /// USD per million input (prompt) tokens.
    pub input_per_million: f64,
    /// USD per million output (completion) tokens.
    pub output_per_million: f64,
    /// USD per million input tokens read from the provider's prompt cache, if listed.
    pub cache_read_per_million: Option<f64>,
    /// USD per million input tokens written to the provider's prompt cache, if listed.
    pub cache_write_per_million: Option<f64>,
    /// Largest prompt the model accepts, in tokens, if listed.
    pub max_input_tokens: Option<u32>,
}

impl ModelPrice {
    /// Cost in USD of a call with the given token counts.
    pub fn cost(&self, input_tokens: u64, output_tokens: u64) -> f64 {
        (input_tokens as f64 * self.input_per_million
            + output_tokens as f64 * self.output_per_million)
            / 1_000_000.0
    }

    /// Cost in USD of a call described by a provider-reported [`Usage`]:
    /// fresh prompt tokens at the input price, cache reads and cache writes at
    /// their own prices (or the input price when the table lists none), and
    /// output tokens at the output price.
    ///
    /// [`Usage`]: crate::usage::Usage
    pub fn cost_usage(&self, usage: &crate::usage::Usage) -> f64 {
        let read = self.cache_read_per_million.unwrap_or(self.input_per_million);
        let write = self.cache_write_per_million.unwrap_or(self.input_per_million);
        (usage.fresh_input_tokens() as f64 * self.input_per_million
            + usage.cached_input_tokens.min(usage.input_tokens) as f64 * read
            + usage.cache_write_tokens.min(usage.input_tokens.saturating_sub(usage.cached_input_tokens)) as f64 * write
            + usage.output_tokens as f64 * self.output_per_million)
            / 1_000_000.0
    }

    /// Cost in USD when part of the prompt was served from the provider's cache.
    ///
    /// `cached_input_tokens` is billed at the cache-read price (or the normal
    /// input price if the table has none) and the rest of `input_tokens` at the
    /// normal input price.
    pub fn cost_with_cache(&self, input_tokens: u64, cached_input_tokens: u64, output_tokens: u64) -> f64 {
        let cached = cached_input_tokens.min(input_tokens);
        let fresh = input_tokens - cached;
        let cache_rate = self.cache_read_per_million.unwrap_or(self.input_per_million);
        (fresh as f64 * self.input_per_million
            + cached as f64 * cache_rate
            + output_tokens as f64 * self.output_per_million)
            / 1_000_000.0
    }
}

/// A lookup table from model name to [`ModelPrice`].
#[derive(Debug, Clone, Default)]
pub struct PriceTable {
    prices: HashMap<String, ModelPrice>,
}

fn opt(v: f64) -> Option<f64> {
    if v.is_nan() { None } else { Some(v) }
}

impl PriceTable {
    /// The table compiled into the crate (see [`SNAPSHOT_DATE`]).
    ///
    /// Built once on first use and shared afterwards.
    pub fn builtin() -> &'static PriceTable {
        static TABLE: OnceLock<PriceTable> = OnceLock::new();
        TABLE.get_or_init(|| {
            let prices = data::ROWS
                .iter()
                .map(|&(model, provider, i, o, cr, cw, max_in)| {
                    (
                        model.to_string(),
                        ModelPrice {
                            model: model.to_string(),
                            provider: provider.to_string(),
                            input_per_million: i,
                            output_per_million: o,
                            cache_read_per_million: opt(cr),
                            cache_write_per_million: opt(cw),
                            max_input_tokens: if max_in == 0 { None } else { Some(max_in) },
                        },
                    )
                })
                .collect();
            PriceTable { prices }
        })
    }

    /// An empty table, for when you want to list your own prices only.
    pub fn new() -> Self {
        Self::default()
    }

    /// Parse a copy of LiteLLM's `model_prices_and_context_window.json`.
    ///
    /// Every entry with `input_cost_per_token` and `output_cost_per_token` is
    /// kept, from every provider. Keys such as `openrouter/x` are stored under
    /// their full name; lookups also try the part after the first `/`.
    ///
    /// # Errors
    /// [`BudgetError::InvalidConfig`] if the text is not a JSON object.
    pub fn from_litellm_json(json: &str) -> Result<Self, BudgetError> {
        let value: serde_json::Value = serde_json::from_str(json)
            .map_err(|e| BudgetError::InvalidConfig(format!("price table JSON: {e}")))?;
        let obj = value
            .as_object()
            .ok_or_else(|| BudgetError::InvalidConfig("price table JSON must be an object".into()))?;
        let mut prices = HashMap::new();
        for (key, v) in obj {
            let f = |name: &str| v.get(name).and_then(serde_json::Value::as_f64);
            let (Some(i), Some(o)) = (f("input_cost_per_token"), f("output_cost_per_token")) else {
                continue;
            };
            let per_m = |x: f64| x * 1_000_000.0;
            prices.insert(
                key.clone(),
                ModelPrice {
                    model: key.clone(),
                    provider: v
                        .get("litellm_provider")
                        .and_then(serde_json::Value::as_str)
                        .unwrap_or("")
                        .to_string(),
                    input_per_million: per_m(i),
                    output_per_million: per_m(o),
                    cache_read_per_million: f("cache_read_input_token_cost").map(per_m),
                    cache_write_per_million: f("cache_creation_input_token_cost").map(per_m),
                    max_input_tokens: v
                        .get("max_input_tokens")
                        .and_then(serde_json::Value::as_u64)
                        .and_then(|n| u32::try_from(n).ok()),
                },
            );
        }
        Ok(Self { prices })
    }

    /// Add or replace one model's price (for private or fine-tuned models).
    pub fn insert(&mut self, price: ModelPrice) {
        self.prices.insert(price.model.clone(), price);
    }

    /// Look up a model. Tries the exact name, then the name without a
    /// `provider/` prefix (so `anthropic/claude-sonnet-4-5` finds `claude-sonnet-4-5`).
    pub fn get(&self, model: &str) -> Option<&ModelPrice> {
        if let Some(p) = self.prices.get(model) {
            return Some(p);
        }
        let (_, bare) = model.split_once('/')?;
        self.prices.get(bare)
    }

    /// Look up a model or return [`BudgetError::UnknownModel`].
    ///
    /// # Errors
    /// [`BudgetError::UnknownModel`] if the model is not in the table.
    pub fn require(&self, model: &str) -> Result<&ModelPrice, BudgetError> {
        self.get(model).ok_or_else(|| BudgetError::UnknownModel(model.to_string()))
    }

    /// Number of models in the table.
    pub fn len(&self) -> usize {
        self.prices.len()
    }

    /// True if the table has no models.
    pub fn is_empty(&self) -> bool {
        self.prices.is_empty()
    }

    /// Iterate over every price in the table (in no particular order).
    pub fn iter(&self) -> impl Iterator<Item = &ModelPrice> {
        self.prices.values()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builtin_rows_are_sorted_and_unique() {
        for w in data::ROWS.windows(2) {
            assert!(w[0].0 < w[1].0, "{} must sort before {}", w[0].0, w[1].0);
        }
    }

    #[test]
    fn builtin_has_main_providers() {
        let t = PriceTable::builtin();
        assert!(t.len() > 100);
        for m in ["gpt-4o", "gpt-4o-mini", "claude-sonnet-4-5", "claude-haiku-4-5"] {
            assert!(t.get(m).is_some(), "missing {m}");
        }
        assert!(t.iter().any(|p| p.provider == "gemini"));
    }

    #[test]
    fn builtin_prices_are_non_negative() {
        for p in PriceTable::builtin().iter() {
            assert!(p.input_per_million >= 0.0 && p.output_per_million >= 0.0, "{}", p.model);
        }
    }

    #[test]
    fn gpt_4o_matches_published_price() {
        // OpenAI lists gpt-4o at $2.50 in / $10.00 out per million tokens.
        let p = PriceTable::builtin().require("gpt-4o").expect("gpt-4o");
        assert_eq!(p.input_per_million, 2.5);
        assert_eq!(p.output_per_million, 10.0);
        assert!((p.cost(1_000_000, 1_000_000) - 12.5).abs() < 1e-9);
    }

    #[test]
    fn provider_prefix_is_stripped() {
        let t = PriceTable::builtin();
        assert_eq!(t.get("openai/gpt-4o"), t.get("gpt-4o"));
        assert!(t.get("nobody/no-such-model").is_none());
    }

    #[test]
    fn unknown_model_is_typed_error() {
        let e = PriceTable::builtin().require("no-such-model").unwrap_err();
        assert_eq!(e, BudgetError::UnknownModel("no-such-model".into()));
    }

    #[test]
    fn cache_reads_are_cheaper() {
        let p = PriceTable::builtin().require("claude-sonnet-4-5").expect("model");
        let full = p.cost(10_000, 0);
        let cached = p.cost_with_cache(10_000, 8_000, 0);
        assert!(cached < full);
        // cached tokens can never exceed input tokens
        assert_eq!(p.cost_with_cache(100, 1_000, 0), p.cost_with_cache(100, 100, 0));
    }

    #[test]
    fn parses_litellm_json() {
        let json = r#"{
            "sample_spec": {"mode": "chat"},
            "my-model": {"input_cost_per_token": 1e-6, "output_cost_per_token": 2e-6,
                         "litellm_provider": "custom", "max_input_tokens": 4096},
            "openrouter/foo": {"input_cost_per_token": 3e-6, "output_cost_per_token": 4e-6}
        }"#;
        let t = PriceTable::from_litellm_json(json).expect("parse");
        assert_eq!(t.len(), 2);
        let p = t.require("my-model").expect("my-model");
        assert!((p.input_per_million - 1.0).abs() < 1e-9);
        assert_eq!(p.max_input_tokens, Some(4096));
        assert!(t.get("openrouter/foo").is_some());
        assert!(PriceTable::from_litellm_json("[1,2]").is_err());
        assert!(PriceTable::from_litellm_json("nope").is_err());
    }

    #[test]
    fn custom_entries_can_be_inserted() {
        let mut t = PriceTable::new();
        assert!(t.is_empty());
        t.insert(ModelPrice {
            model: "in-house".into(),
            provider: "self".into(),
            input_per_million: 0.5,
            output_per_million: 1.0,
            cache_read_per_million: None,
            cache_write_per_million: None,
            max_input_tokens: None,
        });
        assert!((t.require("in-house").expect("x").cost(2_000_000, 0) - 1.0).abs() < 1e-12);
    }
}
