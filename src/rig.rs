//! A dollar cap for [rig](https://crates.io/crates/rig-agent) agents (feature `rig`).
//!
//! [`BudgetHook`] is an [`AgentHook`] for rig-agent 0.44 to 0.46. Attach it to
//! an agent and every model call is authorized against an [`AgentBudget`]
//! before it is sent, and every finished turn is booked at the model's real
//! price (prompt-cache discounts included) after it comes back.
//!
//! ```
//! use std::sync::Arc;
//! use llm_budget::{rig::BudgetHook, AgentBudget, ModelTier};
//!
//! # fn main() -> Result<(), llm_budget::BudgetError> {
//! let budget = Arc::new(AgentBudget::new("researcher", 5.00, ModelTier::Standard, 0.2)?);
//! let hook = BudgetHook::new(budget.clone(), "gpt-4o")
//!     // also hold room for up to 1,000 completion tokens per call
//!     .reserve_output_tokens(1_000);
//! // AgentBuilder::new(model).add_hook(hook).build()
//! # let _ = hook;
//! # Ok(())
//! # }
//! ```
//!
//! A complete, offline run with rig's mock model is in
//! `examples/rig_budget_hook.rs`.
//!
//! # What happens on each call
//!
//! - **`on_completion_call`** (before the provider is called): the hook stops
//!   the run if the budget is killed or spent. Otherwise it estimates the
//!   prompt (history plus the new message, about four characters per token,
//!   plus [`reserve_output_tokens`](BudgetHook::reserve_output_tokens)),
//!   prices it from the [`PriceTable`] and [reserves](AgentBudget::reserve)
//!   that much. If the reservation does not fit in what is left, the run
//!   stops and the call is never made. Reservations are atomic, so agents
//!   sharing one budget cannot all squeeze through on the last dollar.
//! - **`on_model_turn_finished`** (after the provider answered): the
//!   reservation is [settled](crate::Reservation::settle) with the turn's real
//!   cost: the provider's reported cost when rig has one, otherwise the
//!   reported usage priced for `model`. Settlement always books the spend,
//!   even past the cap, because the turn has been paid for. Only then, if the
//!   budget is over its cap, does the hook return `Stop`, and (unless
//!   [`keep_alive_on_refusal`](BudgetHook::keep_alive_on_refusal)) trip the
//!   kill switch so other runs sharing the budget stop before their next call.
//!
//! So [`AgentBudget::spent_usd`] matches what the provider bills, and the cap
//! is overshot by at most what in-flight calls cost beyond their estimates.
//!
//! An attempt rig closes without a model-turn event (for example one it
//! discards and retries) is booked at its estimate when the next call
//! starts, since the provider has most likely charged for it. A reservation
//! still open when a run ends early is released.
//!
//! # How this differs from rig's `run_budget` example
//!
//! rig ships `examples/run_budget.rs`, a hook that caps a single run's
//! **token count** and **tool calls**, with the counters in the run's
//! scratchpad. That is the right tool when tokens are the limit.
//! `BudgetHook` caps **dollars**:
//!
//! - **Across models.** A token is not a unit of money: 100k tokens cost
//!   about $0.015 on gpt-4o-mini and $1.50 on claude-opus. Each turn is
//!   priced for its model, cache reads and writes included, from a table of
//!   232 models (or the provider's own reported cost).
//! - **Across runs and agents.** The budget is an `Arc<AgentBudget>`, not
//!   run-scoped state, so many runs, agents or processes in one service can
//!   share a cap, and a [`FleetAllocator`](crate::FleetAllocator) can split a
//!   fleet total into per-agent caps and move unspent money between them.
//! - **With an audit trail.** Every booked turn is a
//!   [`SpendRecord`](crate::SpendRecord) with model, tokens and cost, and the
//!   budget has a kill switch, a cheaper-tier cascade, and optional `tracing`
//!   and `metrics` events.
//!
//! The two compose: add both hooks to cap tokens per run and dollars overall.

use std::borrow::Borrow;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, PoisonError};

use rig_agent::agent::{
    AgentHook, CompletionCallAction, CompletionCallEvent, HookContext, ModelTurnAction, ModelTurnFinished,
};
use rig_agent::completion::Message;

use crate::budget::{AgentBudget, ChargeContext, Reservation};
use crate::error::BudgetError;
use crate::pricing::PriceTable;
use crate::usage::Usage;

impl From<&rig_agent::completion::Usage> for Usage {
    /// rig reports every prompt token in `input_tokens`, with the cache reads
    /// and writes as parts of it, which is the same shape as [`Usage`].
    fn from(u: &rig_agent::completion::Usage) -> Self {
        let input = u.input_tokens.unwrap_or(0);
        let cached = u.cached_input_tokens.unwrap_or(0).min(input);
        let written = u.cache_creation_input_tokens.unwrap_or(0).min(input - cached);
        Self {
            input_tokens: input,
            cached_input_tokens: cached,
            cache_write_tokens: written,
            output_tokens: u.output_tokens.unwrap_or(0),
        }
    }
}

static NEXT_HOOK_ID: AtomicU64 = AtomicU64::new(0);

/// Open reservations of one run, keyed by hook, kept in the run's scratchpad
/// so they are released when the run is dropped.
#[derive(Clone, Default)]
struct Pending(Arc<Mutex<HashMap<u64, Reservation>>>);

impl Pending {
    fn of(ctx: &HookContext) -> Self {
        if let Some(p) = ctx.scratchpad().get::<Pending>() {
            return p;
        }
        let p = Pending::default();
        ctx.scratchpad().insert(p.clone());
        p
    }

    fn put(&self, hook: u64, r: Reservation) -> Option<Reservation> {
        self.0.lock().unwrap_or_else(PoisonError::into_inner).insert(hook, r)
    }

    fn take(&self, hook: u64) -> Option<Reservation> {
        self.0.lock().unwrap_or_else(PoisonError::into_inner).remove(&hook)
    }
}

/// Authorizes each rig model call against an [`AgentBudget`], books each
/// finished turn, and stops the run at the cap. See the [module docs](self).
pub struct BudgetHook {
    id: u64,
    budget: Arc<AgentBudget>,
    model: String,
    table: Option<PriceTable>,
    kill_on_refusal: bool,
    output_reserve_tokens: u64,
}

impl BudgetHook {
    /// Charge every turn to `budget`, priced as `model` (a name from the
    /// built-in [`PriceTable`], such as `"gpt-4o"` or `"claude-sonnet-4-5"`).
    pub fn new(budget: Arc<AgentBudget>, model: impl Into<String>) -> Self {
        Self {
            id: NEXT_HOOK_ID.fetch_add(1, Ordering::Relaxed),
            budget,
            model: model.into(),
            table: None,
            kill_on_refusal: true,
            output_reserve_tokens: 0,
        }
    }

    /// Price turns from your own table instead of the built-in one.
    pub fn with_price_table(mut self, table: PriceTable) -> Self {
        self.table = Some(table);
        self
    }

    /// Do not trip the kill switch when the cap is reached; only that run stops.
    pub fn keep_alive_on_refusal(mut self) -> Self {
        self.kill_on_refusal = false;
        self
    }

    /// Also reserve `tokens` completion tokens for each call (default 0, so
    /// only the prompt is reserved). Set it to your `max_tokens` to make the
    /// cap hold even for the call that reaches it.
    pub fn reserve_output_tokens(mut self, tokens: u64) -> Self {
        self.output_reserve_tokens = tokens;
        self
    }

    /// The budget this hook charges.
    pub fn budget(&self) -> &Arc<AgentBudget> {
        &self.budget
    }

    fn table(&self) -> &PriceTable {
        match &self.table {
            Some(t) => t,
            None => PriceTable::builtin(),
        }
    }

    /// What the call about to be made is expected to cost. A model missing
    /// from the table estimates $0 (the turn's reported cost is booked later).
    fn estimate_usd<M: Borrow<Message>>(&self, prompt: &Message, history: &[M]) -> f64 {
        let tokens = estimate_prompt_tokens(prompt, history);
        self.table()
            .get(&self.model)
            .map(|p| p.cost(tokens, self.output_reserve_tokens))
            .unwrap_or(0.0)
    }

    fn turn_cost(&self, usage: &rig_agent::completion::Usage) -> Result<f64, BudgetError> {
        match usage.cost.as_ref().map(|c| c.total) {
            Some(total) if total.is_finite() && total >= 0.0 => Ok(total),
            _ => Ok(self.table().require(&self.model)?.cost_usage(&Usage::from(usage))),
        }
    }

    fn stop_reason(&self) -> String {
        format!(
            "Agent '{}' budget exhausted: spent ${:.6}, limit ${:.6}",
            self.budget.agent_id,
            self.budget.spent_usd(),
            self.budget.limit_usd()
        )
    }
}

/// Rough prompt size: about four characters per token over the text of every
/// message. Long unbroken strings (base64 images or files) and raw byte
/// arrays count as 1,000 tokens each, roughly what providers bill for an image.
fn estimate_prompt_tokens<M: Borrow<Message>>(prompt: &Message, history: &[M]) -> u64 {
    fn walk(v: &serde_json::Value) -> u64 {
        use serde_json::Value;
        match v {
            Value::String(s) if s.len() > 4096 && !s.contains(' ') => 1_000,
            Value::String(s) => (s.len() as u64).div_ceil(4),
            Value::Array(a) if a.len() > 256 && a.iter().all(Value::is_number) => 1_000,
            Value::Array(a) => a.iter().map(walk).sum(),
            Value::Object(o) => o.values().map(walk).sum(),
            Value::Number(_) | Value::Bool(_) | Value::Null => 0,
        }
    }
    std::iter::once(prompt)
        .chain(history.iter().map(Borrow::borrow))
        .map(|m| serde_json::to_value(m).map(|v| walk(&v)).unwrap_or(0) + 4)
        .sum()
}

impl AgentHook for BudgetHook {
    fn name(&self) -> Option<String> {
        Some(format!("llm_budget::BudgetHook({}, {})", self.budget.agent_id, self.model))
    }

    async fn on_completion_call(&self, ctx: &HookContext, event: CompletionCallEvent<'_>) -> CompletionCallAction {
        let pending = Pending::of(ctx);
        // An earlier attempt of this run that never produced a model-turn
        // event was most likely billed: book it at its estimate.
        if let Some(stale) = pending.take(self.id) {
            let estimate = stale.estimate_usd();
            stale.settle_with(estimate, ChargeContext::new().model(self.model.clone()));
        }
        if self.budget.is_killed() {
            return CompletionCallAction::Stop(format!("agent '{}' budget is killed", self.budget.agent_id));
        }
        if self.budget.remaining_usd() <= 0.0 {
            return CompletionCallAction::Stop(self.stop_reason());
        }
        let estimate = self.estimate_usd(event.prompt, event.history);
        match self.budget.reserve(estimate) {
            Ok(r) => {
                pending.put(self.id, r);
                CompletionCallAction::Continue
            }
            Err(e) => CompletionCallAction::Stop(format!("{e} (next call estimated at ${estimate:.6})")),
        }
    }

    async fn on_model_turn_finished(&self, ctx: &HookContext, event: ModelTurnFinished<'_>) -> ModelTurnAction {
        let reservation = Pending::of(ctx).take(self.id);
        let cost = match self.turn_cost(&event.usage) {
            Ok(cost) => cost,
            // Without a price nothing can be booked; the reservation is released.
            Err(e) => return ModelTurnAction::Stop(e.to_string()),
        };
        let u = Usage::from(&event.usage);
        let charge = ChargeContext::new().model(self.model.clone()).tokens(u.input_tokens, u.output_tokens);
        let outcome = match reservation {
            Some(r) => r.settle_with(cost, charge),
            None => self.budget.charge_usd_force(cost, charge),
        };
        if !outcome.over_cap {
            return ModelTurnAction::Continue;
        }
        let reason = self.stop_reason();
        if self.kill_on_refusal {
            self.budget.kill(&reason);
        }
        ModelTurnAction::Stop(reason)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rig_agent::completion::Usage as RigUsage;

    #[test]
    fn rig_usage_maps_cache_parts() {
        let u = RigUsage::new().input_tokens(2_100).output_tokens(50).cached_input_tokens(2_000);
        let ours = Usage::from(&u);
        assert_eq!(ours, Usage { input_tokens: 2_100, cached_input_tokens: 2_000, cache_write_tokens: 0, output_tokens: 50 });
        // nonsense counts never underflow
        let odd = RigUsage::new().input_tokens(10).cached_input_tokens(8).cache_creation_input_tokens(8);
        assert_eq!(Usage::from(&odd).fresh_input_tokens(), 0);
    }

    #[test]
    fn prompt_estimate_grows_with_text_and_caps_blobs() {
        let short = Message::user("hi");
        let long = Message::user("word ".repeat(4_000));
        let none: [Message; 0] = [];
        let a = estimate_prompt_tokens(&short, &none);
        let b = estimate_prompt_tokens(&long, &none);
        assert!(a < 20, "{a}");
        assert!((5_000..5_200).contains(&b), "{b}");
        let blob = Message::user("A".repeat(1_000_000));
        assert!(estimate_prompt_tokens(&blob, &none) <= 1_100);
        let history = [Arc::new(long.clone()), Arc::new(short.clone())];
        assert!(estimate_prompt_tokens(&short, &history) > b);
    }
}
