//! Per-model prices for [borg](https://github.com/joydle/borg) budgets
//! (feature `borg`).
//!
//! borg's `Budget` trait counts tokens, and the executor, `SubBudget` and
//! `BudgetExhausted` all assume token units. A token on a frontier model can
//! cost 100x a token on a small one, so a plain token cap misprices a tree
//! that mixes models. This module keeps borg's units and prices each call by
//! the model it actually used.
//!
//! # Units: reference tokens
//!
//! You pick one *reference model*, normally the main model of the tree. A
//! call is charged the number of reference-model tokens that cost the same,
//! at that call's own input/output mix:
//!
//! ```text
//! charged = tokens_in_call * cost(call's model, usage) / cost(reference, same tokens, no cache)
//! ```
//!
//! - A call on the reference model with no cache hits is charged exactly its
//!   token count, the same as borg's `InMemoryBudget`.
//! - A call on a model that costs 10x as much for that mix is charged 10x
//!   its tokens; a 20x cheaper model is charged 1/20th.
//! - Prompt-cache reads and writes are priced at the model's cache prices,
//!   so a cached call is charged less (or, for Anthropic cache writes, more).
//!
//! Every limit, `remaining()`, `admit_parallel(n, 1_000)` and
//! `BudgetExhausted { needed, remaining }` stays in tokens (reference
//! tokens), so borg's parallel admission and `SubBudget` keep working.
//! Size the session budget as "this many tokens of the reference model".
//!
//! # How the model reaches the budget
//!
//! borg's `TokenUsage` does not say which model a call used, and the trait
//! has no model on the charge path. borg's own charge site (the default
//! `BorgAgent::invoke`) calls `ctx.llm.chat_completion(request)` and then,
//! in the same poll with no await in between, `ctx.budget.record_usage(..)`.
//! So the adapter wraps both:
//!
//! - `PricedLlm` wraps your `AgentLlmClient`. When a response comes back it
//!   notes `(request.model, usage)` for the current thread on the shared
//!   [`BorgPricing`].
//! - `PricedBudget` wraps any `Arc<dyn Budget>` (`InMemoryBudget`,
//!   `SubBudget`, your own). `record_usage` takes the note left on its thread,
//!   checks the usage matches, prices the call on that model and charges the
//!   inner budget that many reference tokens with `record`.
//!
//! The note lives on the [`BorgPricing`] you create (no process-wide state)
//! and is consumed by the charge. A charge with no matching note (a custom
//! `invoke` that records somewhere else, a client not wrapped in
//! `PricedLlm`) is priced as the reference model and counted in
//! [`BorgPricing::unattributed_calls`], so you can see it happen.
//!
//! # Using it
//!
//! llm-budget does not depend on borg (borg is not on crates.io). The
//! [`borg_adapter!`](crate::borg_adapter) macro writes the two small adapter
//! types into your crate, against the borg version your crate already uses:
//!
//! ```toml
//! [dependencies]
//! borg = { git = "https://github.com/joydle/borg", tag = "v1.0.0" }
//! llm-budget = { version = "0.9", default-features = false, features = ["borg"] }
//! ```
//!
//! ```ignore
//! llm_budget::borg_adapter!(); // defines PricedBudget and PricedLlm here
//!
//! use std::sync::Arc;
//! use llm_budget::borg::BorgPricing;
//!
//! // Budgets are in claude-sonnet-4-6 tokens; every call is priced on its own model.
//! let pricing = Arc::new(BorgPricing::new("claude-sonnet-4-6")?);
//! let llm = PricedLlm::shared(&pricing, my_llm_client);
//! let session = PricedBudget::in_memory(&pricing, 200_000);
//! let ctx = borg::ExecCtx::new(llm, session.clone(), emitter, cancel, "session");
//!
//! // A pool for one composition, still capped by the session:
//! let round = PricedBudget::sub(&pricing, session.clone(), 50_000);
//!
//! println!("${:.4} spent", pricing.spent_usd());
//! ```
//!
//! Keep a `PricedBudget` outermost on whatever budget goes into an
//! `ExecCtx`. `PricedBudget::sub` does that for you; a bare
//! `borg::sub_budget(priced_session, n)` would count its own pool in raw
//! tokens.
//!
//! Guide with the unit table and rules:
//! [docs/borg.md](https://github.com/Mattbusel/llm-budget/blob/main/docs/borg.md).
//!
//! The price table is shared behind an `Arc`
//! ([`BorgPricing::with_table`]); [`BorgPricing::new`] uses the built-in
//! table without copying it.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::ThreadId;

use crate::error::BudgetError;
use crate::pricing::PriceTable;
use crate::usage::Usage;

/// Nano-dollars per dollar, for the spend counter.
const NANOS: f64 = 1_000_000_000.0;

#[derive(Clone)]
enum Table {
    Builtin,
    Shared(Arc<PriceTable>),
}

impl Table {
    fn get(&self) -> &PriceTable {
        match self {
            Table::Builtin => PriceTable::builtin(),
            Table::Shared(t) => t,
        }
    }
}

struct Note {
    thread: ThreadId,
    model: String,
    usage: Usage,
}

/// Prices borg calls by model, in reference-model tokens. Shared by the
/// `PricedLlm` and every `PricedBudget` that [`borg_adapter!`](crate::borg_adapter)
/// defines; see the [module docs](self).
pub struct BorgPricing {
    table: Table,
    reference: String,
    unpriced_ratio: f64,
    notes: Mutex<Vec<Note>>,
    spent_nanos: AtomicU64,
    unpriced_calls: AtomicU64,
    unattributed_calls: AtomicU64,
}

impl std::fmt::Debug for BorgPricing {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BorgPricing")
            .field("reference", &self.reference)
            .field("unpriced_ratio", &self.unpriced_ratio)
            .field("spent_usd", &self.spent_usd())
            .field("unpriced_calls", &self.unpriced_calls())
            .field("unattributed_calls", &self.unattributed_calls())
            .finish_non_exhaustive()
    }
}

impl BorgPricing {
    /// Price with the built-in table ([`PriceTable::builtin`], not copied),
    /// in tokens of `reference_model`.
    ///
    /// ```
    /// let p = llm_budget::borg::BorgPricing::new("claude-sonnet-4-6")?;
    /// assert_eq!(p.reference_model(), "claude-sonnet-4-6");
    /// # Ok::<(), llm_budget::BudgetError>(())
    /// ```
    ///
    /// # Errors
    /// [`BudgetError::UnknownModel`] if the table cannot price the reference
    /// model, [`BudgetError::InvalidConfig`] if its price is zero.
    pub fn new(reference_model: impl Into<String>) -> Result<Self, BudgetError> {
        Self::build(Table::Builtin, reference_model.into())
    }

    /// Price with your own table, shared behind an `Arc` (one table can serve
    /// many pricings and budgets without being copied).
    ///
    /// # Errors
    /// As [`new`](Self::new).
    pub fn with_table(table: Arc<PriceTable>, reference_model: impl Into<String>) -> Result<Self, BudgetError> {
        Self::build(Table::Shared(table), reference_model.into())
    }

    fn build(table: Table, reference: String) -> Result<Self, BudgetError> {
        let probe = table.get().cost_usage(&reference, &Usage::new(1_000_000, 1_000_000))?;
        if !(probe.is_finite() && probe > 0.0) {
            return Err(BudgetError::InvalidConfig(format!(
                "reference model {reference} has no positive price"
            )));
        }
        Ok(Self {
            table,
            reference,
            unpriced_ratio: 1.0,
            notes: Mutex::new(Vec::new()),
            spent_nanos: AtomicU64::new(0),
            unpriced_calls: AtomicU64::new(0),
            unattributed_calls: AtomicU64::new(0),
        })
    }

    /// How to charge a call whose model the table cannot price: its tokens
    /// times `ratio` (default `1.0`, the same as a reference-model call).
    /// Negative or non-finite values are treated as `1.0`. Such calls are
    /// counted in [`unpriced_calls`](Self::unpriced_calls) and left out of
    /// [`spent_usd`](Self::spent_usd).
    #[must_use]
    pub fn unpriced_ratio(mut self, ratio: f64) -> Self {
        self.unpriced_ratio = if ratio.is_finite() && ratio >= 0.0 { ratio } else { 1.0 };
        self
    }

    /// The model budgets are denominated in.
    pub fn reference_model(&self) -> &str {
        &self.reference
    }

    /// The price table in use.
    pub fn table(&self) -> &PriceTable {
        self.table.get()
    }

    /// Reference tokens a call to `model` with `usage` costs. Pure: books
    /// nothing and touches no counters.
    ///
    /// ```
    /// use std::sync::Arc;
    /// use llm_budget::{borg::BorgPricing, ModelPrice, PriceTable, Usage};
    ///
    /// let table = PriceTable::new()
    ///     .with_price(ModelPrice::new("small", "x", 1.0, 5.0))
    ///     .with_price(ModelPrice::new("big", "x", 10.0, 50.0));
    /// let p = BorgPricing::with_table(Arc::new(table), "small")?;
    /// let u = Usage::new(1_000, 200);
    /// assert_eq!(p.units("small", &u), 1_200); // the reference: its own tokens
    /// assert_eq!(p.units("big", &u), 12_000); // 10x the price, 10x the charge
    /// # Ok::<(), llm_budget::BudgetError>(())
    /// ```
    pub fn units(&self, model: &str, usage: &Usage) -> u64 {
        self.price(Some(model), usage).0
    }

    /// (units, priced USD or None when unpriced)
    fn price(&self, model: Option<&str>, usage: &Usage) -> (u64, Option<f64>) {
        let tokens = usage.total_tokens();
        if tokens == 0 {
            return (0, Some(0.0));
        }
        let table = self.table.get();
        let model = model.unwrap_or(&self.reference);
        let reference = table
            .cost_usage(&self.reference, &Usage::new(usage.input_tokens, usage.output_tokens))
            .ok()
            .filter(|c| c.is_finite() && *c > 0.0);
        let cost = table.cost_usage(model, usage).ok().filter(|c| c.is_finite() && *c >= 0.0);
        match (cost, reference) {
            (Some(cost), Some(reference)) => (to_units(tokens as f64 * (cost / reference)), Some(cost)),
            _ => (to_units(tokens as f64 * self.unpriced_ratio), None),
        }
    }

    /// Note that a call to `model` just returned `usage` on this thread. The
    /// next [`charge`](Self::charge) on the same thread with the same usage is
    /// priced on `model`. `PricedLlm` calls this; call it yourself only from
    /// an LLM client of your own that the adapter does not wrap.
    pub fn note_call(&self, model: &str, usage: Usage) {
        let thread = std::thread::current().id();
        let mut notes = self.notes.lock().unwrap_or_else(|e| e.into_inner());
        notes.retain(|n| n.thread != thread);
        notes.push(Note { thread, model: model.to_owned(), usage });
    }

    /// Price a finished call with `usage` and return the reference tokens to
    /// charge. Takes this thread's [`note_call`](Self::note_call) if its usage
    /// matches; otherwise prices the call as the reference model and counts
    /// it in [`unattributed_calls`](Self::unattributed_calls). Adds the priced
    /// cost to [`spent_usd`](Self::spent_usd). `PricedBudget::record_usage`
    /// calls this.
    pub fn charge(&self, usage: &Usage) -> u64 {
        let model = self.take_note(usage);
        if model.is_none() && !usage.is_empty() {
            self.unattributed_calls.fetch_add(1, Ordering::Relaxed);
        }
        let (units, usd) = self.price(model.as_deref(), usage);
        match usd {
            Some(usd) => {
                let nanos = (usd * NANOS).round();
                let nanos = if nanos >= u64::MAX as f64 { u64::MAX } else { nanos as u64 };
                saturating_add(&self.spent_nanos, nanos);
            }
            None => {
                self.unpriced_calls.fetch_add(1, Ordering::Relaxed);
            }
        }
        units
    }

    fn take_note(&self, usage: &Usage) -> Option<String> {
        let thread = std::thread::current().id();
        let mut notes = self.notes.lock().unwrap_or_else(|e| e.into_inner());
        let i = notes.iter().position(|n| n.thread == thread)?;
        let note = notes.swap_remove(i);
        (note.usage == *usage).then_some(note.model)
    }

    /// USD spent on priced calls charged through this pricing, across every
    /// budget made from it.
    pub fn spent_usd(&self) -> f64 {
        self.spent_nanos.load(Ordering::Relaxed) as f64 / NANOS
    }

    /// Calls whose model the table could not price (charged with
    /// [`unpriced_ratio`](Self::unpriced_ratio)).
    pub fn unpriced_calls(&self) -> u64 {
        self.unpriced_calls.load(Ordering::Relaxed)
    }

    /// Charges that arrived with no matching note from `PricedLlm`, priced as
    /// the reference model. Non-zero means a call path the adapter does not
    /// see; see the [module docs](self).
    pub fn unattributed_calls(&self) -> u64 {
        self.unattributed_calls.load(Ordering::Relaxed)
    }
}

fn to_units(x: f64) -> u64 {
    if x.is_nan() || x <= 0.0 {
        0
    } else if x >= u64::MAX as f64 {
        u64::MAX
    } else {
        x.round() as u64
    }
}

fn saturating_add(a: &AtomicU64, n: u64) {
    let mut cur = a.load(Ordering::Relaxed);
    loop {
        match a.compare_exchange_weak(cur, cur.saturating_add(n), Ordering::Relaxed, Ordering::Relaxed) {
            Ok(_) => return,
            Err(c) => cur = c,
        }
    }
}

/// Converts borg's usage fields (input including the cache split, output,
/// cache reads, cache writes) to a [`Usage`]. Used by the macro.
#[doc(hidden)]
pub fn __usage(input: u64, output: u64, cache_read: u64, cache_creation: u64) -> Usage {
    let read = cache_read.min(input);
    let write = cache_creation.min(input - read);
    Usage { input_tokens: input, cached_input_tokens: read, cache_write_tokens: write, output_tokens: output }
}

#[doc(hidden)]
pub mod __private {
    pub use async_trait::async_trait;
}

/// Writes `PricedBudget` and `PricedLlm`, adapters that implement borg's
/// `Budget` and `AgentLlmClient` traits, into the calling module (feature
/// `borg`). Pass a visibility for the generated items (default `pub`):
/// `llm_budget::borg_adapter!(pub(crate));`.
///
/// The code names `::borg`, so the calling crate must depend on borg under
/// that name. It needs nothing else: `async-trait` comes from llm-budget.
/// See [`borg`](crate::borg) for units, attribution and an example.
///
/// Generated API:
///
/// - `PricedBudget::new(&Arc<BorgPricing>, Arc<dyn borg::Budget>) -> PricedBudget`
/// - `PricedBudget::shared(&pricing, inner) -> Arc<dyn borg::Budget>`
/// - `PricedBudget::in_memory(&pricing, limit) -> Arc<dyn borg::Budget>`: a
///   fresh `borg::InMemoryBudget` of `limit` reference tokens.
/// - `PricedBudget::sub(&pricing, parent, limit) -> Arc<dyn borg::Budget>`: a
///   `borg::SubBudget` of `limit` inside `parent`, priced.
/// - `PricedBudget::inner(&self) -> &Arc<dyn borg::Budget>`
/// - `PricedLlm::new(&pricing, Arc<dyn borg::AgentLlmClient>) -> PricedLlm`
/// - `PricedLlm::shared(&pricing, client) -> Arc<dyn borg::AgentLlmClient>`
#[macro_export]
macro_rules! borg_adapter {
    () => {
        $crate::borg_adapter!(pub);
    };
    ($vis:vis) => {
        /// A borg `Budget` that charges each call in reference-model tokens,
        /// priced on the model the call used (llm-budget, feature `borg`).
        $vis struct PricedBudget {
            inner: ::std::sync::Arc<dyn ::borg::Budget>,
            pricing: ::std::sync::Arc<$crate::borg::BorgPricing>,
        }

        #[allow(dead_code)]
        impl PricedBudget {
            /// Wrap `inner`, whose limit is in reference-model tokens.
            $vis fn new(
                pricing: &::std::sync::Arc<$crate::borg::BorgPricing>,
                inner: ::std::sync::Arc<dyn ::borg::Budget>,
            ) -> Self {
                Self { inner, pricing: ::std::sync::Arc::clone(pricing) }
            }

            /// [`PricedBudget::new`] as an `Arc<dyn borg::Budget>`.
            $vis fn shared(
                pricing: &::std::sync::Arc<$crate::borg::BorgPricing>,
                inner: ::std::sync::Arc<dyn ::borg::Budget>,
            ) -> ::std::sync::Arc<dyn ::borg::Budget> {
                ::std::sync::Arc::new(Self::new(pricing, inner))
            }

            /// A new budget of `limit` reference-model tokens.
            $vis fn in_memory(
                pricing: &::std::sync::Arc<$crate::borg::BorgPricing>,
                limit: u64,
            ) -> ::std::sync::Arc<dyn ::borg::Budget> {
                Self::shared(pricing, ::std::sync::Arc::new(::borg::InMemoryBudget::new(limit)))
            }

            /// A pool of `limit` reference-model tokens inside `parent`
            /// (`borg::SubBudget`), priced. Spent when either is.
            $vis fn sub(
                pricing: &::std::sync::Arc<$crate::borg::BorgPricing>,
                parent: ::std::sync::Arc<dyn ::borg::Budget>,
                limit: u64,
            ) -> ::std::sync::Arc<dyn ::borg::Budget> {
                Self::shared(pricing, ::borg::sub_budget(parent, limit))
            }

            /// The wrapped budget.
            $vis fn inner(&self) -> &::std::sync::Arc<dyn ::borg::Budget> {
                &self.inner
            }
        }

        impl ::borg::Budget for PricedBudget {
            fn consumed(&self) -> u64 {
                self.inner.consumed()
            }
            fn remaining(&self) -> u64 {
                self.inner.remaining()
            }
            fn limit(&self) -> u64 {
                self.inner.limit()
            }
            fn can_invoke(&self) -> bool {
                self.inner.can_invoke()
            }
            /// `tokens` are already reference tokens (a priced child
            /// charging its parent): passed through unchanged.
            fn record(&self, tokens: u64) -> ::borg::BudgetVerdict {
                self.inner.record(tokens)
            }
            fn record_usage(&self, usage: &::borg::TokenUsage) -> ::borg::BudgetVerdict {
                let usage = $crate::borg::__usage(
                    usage.input_tokens,
                    usage.output_tokens,
                    usage.cache_read_tokens,
                    usage.cache_creation_tokens,
                );
                self.inner.record(self.pricing.charge(&usage))
            }
            fn admit_parallel(
                &self,
                branches: usize,
                min_per_branch: u64,
            ) -> ::borg::budget::ParallelAdmission {
                self.inner.admit_parallel(branches, min_per_branch)
            }
        }

        /// A borg `AgentLlmClient` that tells the shared `BorgPricing` which
        /// model each call used (llm-budget, feature `borg`).
        $vis struct PricedLlm {
            inner: ::std::sync::Arc<dyn ::borg::AgentLlmClient>,
            pricing: ::std::sync::Arc<$crate::borg::BorgPricing>,
        }

        #[allow(dead_code)]
        impl PricedLlm {
            /// Wrap `inner`.
            $vis fn new(
                pricing: &::std::sync::Arc<$crate::borg::BorgPricing>,
                inner: ::std::sync::Arc<dyn ::borg::AgentLlmClient>,
            ) -> Self {
                Self { inner, pricing: ::std::sync::Arc::clone(pricing) }
            }

            /// [`PricedLlm::new`] as an `Arc<dyn borg::AgentLlmClient>`.
            $vis fn shared(
                pricing: &::std::sync::Arc<$crate::borg::BorgPricing>,
                inner: ::std::sync::Arc<dyn ::borg::AgentLlmClient>,
            ) -> ::std::sync::Arc<dyn ::borg::AgentLlmClient> {
                ::std::sync::Arc::new(Self::new(pricing, inner))
            }
        }

        #[$crate::borg::__private::async_trait]
        impl ::borg::AgentLlmClient for PricedLlm {
            async fn chat_completion(
                &self,
                request: ::borg::ChatRequest,
            ) -> ::std::result::Result<::borg::ChatResponse, ::borg::BorgError> {
                let model = ::std::clone::Clone::clone(&request.model);
                let response = self.inner.chat_completion(request).await?;
                // No await between here and the agent's record_usage.
                self.pricing.note_call(
                    &model,
                    $crate::borg::__usage(
                        response.input_tokens,
                        response.output_tokens,
                        response.cache_read_tokens,
                        response.cache_creation_tokens,
                    ),
                );
                ::std::result::Result::Ok(response)
            }
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pricing::ModelPrice;

    fn pricing() -> BorgPricing {
        let table = PriceTable::new()
            .with_price(ModelPrice::new("small", "x", 1.0, 5.0).with_cache(Some(0.1), Some(1.25)))
            .with_price(ModelPrice::new("big", "x", 15.0, 75.0));
        match BorgPricing::with_table(Arc::new(table), "small") {
            Ok(p) => p,
            Err(e) => unreachable!("{e}"),
        }
    }

    #[test]
    fn reference_calls_charge_their_tokens() {
        let p = pricing();
        for (i, o) in [(1, 0), (0, 1), (1_000, 200), (123_457, 9_999)] {
            assert_eq!(p.units("small", &Usage::new(i, o)), i + o);
        }
        assert_eq!(p.units("small", &Usage::default()), 0);
    }

    #[test]
    fn other_models_charge_by_price() {
        let p = pricing();
        assert_eq!(p.units("big", &Usage::new(1_000, 1_000)), 30_000);
        // Cache reads at a tenth of the input price.
        let cached = __usage(10_000, 0, 9_000, 0);
        assert_eq!(p.units("small", &cached), 1_000 + 900);
    }

    #[test]
    fn charge_uses_this_threads_note_once() {
        let p = pricing();
        let u = Usage::new(1_000, 1_000);
        p.note_call("big", u);
        assert_eq!(p.charge(&u), 30_000);
        assert_eq!(p.charge(&u), 2_000, "the note is consumed");
        assert_eq!(p.unattributed_calls(), 1);
        assert!((p.spent_usd() - (0.09 + 0.006)).abs() < 1e-9);
    }

    #[test]
    fn a_note_with_other_usage_is_not_used() {
        let p = pricing();
        p.note_call("big", Usage::new(5, 5));
        assert_eq!(p.charge(&Usage::new(1_000, 1_000)), 2_000);
        assert_eq!(p.unattributed_calls(), 1);
    }

    #[test]
    fn notes_do_not_cross_threads() {
        let p = Arc::new(pricing());
        let u = Usage::new(1_000, 1_000);
        p.note_call("big", u);
        let q = Arc::clone(&p);
        let other = std::thread::spawn(move || q.charge(&u)).join();
        assert_eq!(other.ok(), Some(2_000));
        assert_eq!(p.charge(&u), 30_000);
    }

    #[test]
    fn unpriced_models_use_the_ratio() {
        let p = pricing().unpriced_ratio(3.0);
        let u = Usage::new(100, 0);
        p.note_call("mystery", u);
        assert_eq!(p.charge(&u), 300);
        assert_eq!(p.unpriced_calls(), 1);
        assert_eq!(p.spent_usd(), 0.0);
    }

    #[test]
    fn reference_must_be_priced() {
        let t = Arc::new(PriceTable::new().with_price(ModelPrice::new("free", "x", 0.0, 0.0)));
        assert!(matches!(BorgPricing::with_table(t.clone(), "free"), Err(BudgetError::InvalidConfig(_))));
        assert!(matches!(BorgPricing::with_table(t, "nope"), Err(BudgetError::UnknownModel(_))));
    }

    #[test]
    fn builtin_table_prices_real_models() {
        let p = match BorgPricing::new("claude-sonnet-4-6") {
            Ok(p) => p,
            Err(e) => unreachable!("{e}"),
        };
        let u = Usage::new(10_000, 1_000);
        assert_eq!(p.units("claude-sonnet-4-6", &u), 11_000);
        assert!(p.units("claude-haiku-4-5", &u) < 11_000);
    }

    #[test]
    fn huge_usage_saturates() {
        let p = pricing();
        assert_eq!(p.units("big", &Usage::new(u64::MAX, u64::MAX)), u64::MAX);
    }
}
