//! A dollar cap for [rig](https://crates.io/crates/rig-agent) agents (feature `rig`).
//!
//! [`BudgetHook`] is an [`AgentHook`] for rig-agent 0.44. Attach it to an
//! agent and every model turn is charged to an [`AgentBudget`] at the model's
//! real price, prompt-cache discounts included. The run stops with a clear
//! reason the moment a turn would cross the cap, and a run on a budget that is
//! already spent or killed stops before its first paid call.
//!
//! ```
//! use std::sync::Arc;
//! use llm_budget::{rig::BudgetHook, AgentBudget, ModelTier};
//! use rig_agent::agent::AgentBuilder;
//! use rig_agent::core::completion::Usage;
//! use rig_agent::test_utils::{MockAddTool, MockCompletionModel, MockTurn};
//!
//! # #[tokio::main(flavor = "current_thread")]
//! # async fn main() -> Result<(), llm_budget::BudgetError> {
//! // $0.05 for gpt-4o. Each turn below is 10,000 prompt + 500 completion tokens = $0.03.
//! let budget = Arc::new(AgentBudget::new("researcher", 0.05, ModelTier::Standard, 0.2)?);
//! let turn = |id: &str| {
//!     MockTurn::tool_call(id, "add", serde_json::json!({"x": 1, "y": 1}))
//!         .with_usage(Usage::new().input_tokens(10_000).output_tokens(500))
//! };
//! let model = MockCompletionModel::from_turns([turn("a"), turn("b"), turn("c"), MockTurn::text("done")]);
//!
//! let agent = AgentBuilder::new(model)
//!     .tool(MockAddTool)
//!     .add_hook(BudgetHook::new(budget.clone(), "gpt-4o"))
//!     .build();
//! let err = agent.prompt("go").max_turns(10).await.unwrap_err();
//!
//! assert!(err.to_string().contains("budget exhausted"));
//! assert!((budget.spent_usd() - 0.03).abs() < 1e-9); // the first turn; the second was refused
//! # Ok(())
//! # }
//! ```
//!
//! Share one `Arc<AgentBudget>` between several agents or runs to give them a
//! common cap. For a fleet, register the agent in a
//! [`FleetAllocator`](crate::FleetAllocator) and pass its budget here.

use std::sync::Arc;

use rig_agent::agent::{
    AgentHook, CompletionCallAction, CompletionCallEvent, HookContext, ModelTurnAction, ModelTurnFinished,
};

use crate::budget::AgentBudget;
use crate::error::BudgetError;
use crate::pricing::PriceTable;
use crate::usage::Usage;

impl From<&rig_agent::core::completion::Usage> for Usage {
    /// rig reports every prompt token in `input_tokens`, with the cache reads
    /// and writes as parts of it, which is the same shape as [`Usage`].
    fn from(u: &rig_agent::core::completion::Usage) -> Self {
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

/// Charges each rig model turn to an [`AgentBudget`] and stops the run at the cap.
///
/// What it does at each point of a run:
/// - **before a model call**: if the budget is killed or has nothing left, the
///   run stops and the provider is never called.
/// - **after a model turn**: the turn is charged. When rig reports a cost for
///   the turn (some providers do) that figure is charged; otherwise the
///   reported usage is priced for `model` from the price table. If the charge
///   would cross the cap it is refused and the run stops with the
///   [`BudgetError`] text, for example
///   `Agent 'researcher' budget exhausted: spent $0.030000, limit $0.050000`.
///
/// The refused turn has already been paid to the provider, so the hook also
/// trips the budget's kill switch: any later run sharing it stops before its
/// first call instead of paying for one more turn and being refused again.
/// Turn that off with [`keep_alive_on_refusal`](Self::keep_alive_on_refusal)
/// if you raise caps at runtime.
pub struct BudgetHook {
    budget: Arc<AgentBudget>,
    model: String,
    table: Option<PriceTable>,
    kill_on_refusal: bool,
}

impl BudgetHook {
    /// Charge every turn to `budget`, priced as `model` (a name from the
    /// built-in [`PriceTable`], such as `"gpt-4o"` or `"claude-sonnet-4-5"`).
    pub fn new(budget: Arc<AgentBudget>, model: impl Into<String>) -> Self {
        Self { budget, model: model.into(), table: None, kill_on_refusal: true }
    }

    /// Price turns from your own table instead of the built-in one.
    pub fn with_price_table(mut self, table: PriceTable) -> Self {
        self.table = Some(table);
        self
    }

    /// Do not trip the kill switch when a turn is refused; only that run stops.
    pub fn keep_alive_on_refusal(mut self) -> Self {
        self.kill_on_refusal = false;
        self
    }

    /// The budget this hook charges.
    pub fn budget(&self) -> &Arc<AgentBudget> {
        &self.budget
    }

    async fn charge(&self, usage: &rig_agent::core::completion::Usage) -> Result<(), BudgetError> {
        let u = Usage::from(usage);
        let cost = match usage.cost.as_ref().map(|c| c.total) {
            Some(total) if total.is_finite() && total >= 0.0 => total,
            _ => {
                let table: &PriceTable = match &self.table {
                    Some(t) => t,
                    None => PriceTable::builtin(),
                };
                table.require(&self.model)?.cost_usage(&u)
            }
        };
        let clamp = |n: u64| u32::try_from(n).unwrap_or(u32::MAX);
        self.budget
            .charge_inner(Some(&self.model), Some(cost), clamp(u.input_tokens), clamp(u.output_tokens), None, None)
            .await
            .map(|_| ())
    }
}

impl AgentHook for BudgetHook {
    fn name(&self) -> Option<String> {
        Some(format!("llm_budget::BudgetHook({}, {})", self.budget.agent_id, self.model))
    }

    async fn on_completion_call(&self, _ctx: &HookContext, _event: CompletionCallEvent<'_>) -> CompletionCallAction {
        if self.budget.is_killed() {
            return CompletionCallAction::Stop(format!("agent '{}' budget is killed", self.budget.agent_id));
        }
        if self.budget.remaining_usd() <= 0.0 {
            return CompletionCallAction::Stop(format!(
                "agent '{}' budget exhausted: ${:.6} of ${:.6} spent",
                self.budget.agent_id,
                self.budget.spent_usd(),
                self.budget.limit_usd()
            ));
        }
        CompletionCallAction::Continue
    }

    async fn on_model_turn_finished(&self, _ctx: &HookContext, event: ModelTurnFinished<'_>) -> ModelTurnAction {
        match self.charge(&event.usage).await {
            Ok(()) => ModelTurnAction::Continue,
            Err(e) => {
                let reason = e.to_string();
                if self.kill_on_refusal && !matches!(e, BudgetError::UnknownModel(_)) {
                    self.budget.kill(&reason);
                }
                ModelTurnAction::Stop(reason)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::ModelTier;
    use rig_agent::agent::AgentBuilder;
    use rig_agent::core::completion::Usage as RigUsage;
    use rig_agent::test_utils::{MockAddTool, MockCompletionModel, MockTurn};
    use serde_json::json;

    fn tool_turn(id: &str, usage: RigUsage) -> MockTurn {
        MockTurn::tool_call(id, "add", json!({"x": 1, "y": 1})).with_usage(usage)
    }

    fn budget(limit: f64) -> Arc<AgentBudget> {
        Arc::new(AgentBudget::new("researcher", limit, ModelTier::Standard, 0.2).unwrap())
    }

    #[test]
    fn rig_usage_maps_cache_parts() {
        let u = RigUsage::new().input_tokens(2_100).output_tokens(50).cached_input_tokens(2_000);
        let ours = Usage::from(&u);
        assert_eq!(ours, Usage { input_tokens: 2_100, cached_input_tokens: 2_000, cache_write_tokens: 0, output_tokens: 50 });
        // nonsense counts never underflow
        let odd = RigUsage::new().input_tokens(10).cached_input_tokens(8).cache_creation_input_tokens(8);
        assert_eq!(Usage::from(&odd).fresh_input_tokens(), 0);
    }

    #[tokio::test]
    async fn stops_the_run_at_the_cap_and_kills_the_budget() {
        let b = budget(0.05);
        let t = || RigUsage::new().input_tokens(10_000).output_tokens(500);
        let model = MockCompletionModel::from_turns([
            tool_turn("a", t()),
            tool_turn("b", t()),
            tool_turn("c", t()),
            MockTurn::text("done"),
        ]);
        let err = AgentBuilder::new(model)
            .tool(MockAddTool)
            .add_hook(BudgetHook::new(b.clone(), "gpt-4o"))
            .build()
            .prompt("go")
            .max_turns(10)
            .await
            .unwrap_err();
        assert!(err.to_string().contains("budget exhausted"), "{err}");
        assert!((b.spent_usd() - 0.03).abs() < 1e-9, "spent {}", b.spent_usd());
        assert!(b.is_killed());
    }

    #[tokio::test]
    async fn a_killed_budget_stops_before_the_first_call() {
        let b = budget(1.0);
        b.kill("manual stop");
        let model = MockCompletionModel::from_turns([MockTurn::text("never sent")]);
        let err = AgentBuilder::new(model)
            .add_hook(BudgetHook::new(b.clone(), "gpt-4o"))
            .build()
            .prompt("go")
            .await
            .unwrap_err();
        assert!(err.to_string().contains("killed"), "{err}");
        assert_eq!(b.spent_usd(), 0.0);
    }

    #[tokio::test]
    async fn cache_reads_are_charged_at_the_cache_price() {
        let b = budget(1.0);
        let usage = RigUsage::new().input_tokens(2_100).output_tokens(50).cached_input_tokens(2_000);
        let model = MockCompletionModel::from_turns([MockTurn::text("ok").with_usage(usage)]);
        AgentBuilder::new(model)
            .add_hook(BudgetHook::new(b.clone(), "gpt-4o"))
            .build()
            .prompt("go")
            .await
            .unwrap();
        let expected = PriceTable::builtin().require("gpt-4o").unwrap().cost_usage(&Usage::from(&usage));
        let full = PriceTable::builtin().require("gpt-4o").unwrap().cost(2_100, 50);
        assert!((b.spent_usd() - expected).abs() < 1e-9);
        assert!(b.spent_usd() < full);
    }

    #[tokio::test]
    async fn a_provider_reported_cost_wins_over_the_table() {
        let b = budget(1.0);
        let usage = RigUsage::new()
            .input_tokens(1_000)
            .output_tokens(100)
            .cost(rig_agent::core::completion::Cost::from_parts(0.01, 0.002, 0.0, 0.0));
        let model = MockCompletionModel::from_turns([MockTurn::text("ok").with_usage(usage)]);
        AgentBuilder::new(model)
            .add_hook(BudgetHook::new(b.clone(), "some-model-not-in-the-table"))
            .build()
            .prompt("go")
            .await
            .unwrap();
        assert!((b.spent_usd() - 0.012).abs() < 1e-9);
    }
}
