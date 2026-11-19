//! `rig::BudgetHook` against rig's mock model.
//!
//! Needs rig-agent's test utilities:
//! `cargo test --features __rig-test-utils --test rig_hook` (or `--all-features`).

#![allow(clippy::unwrap_used)]

use std::sync::Arc;

use llm_budget::rig::BudgetHook;
use llm_budget::{AgentBudget, AuditLog, ModelTier, PriceTable, Usage};
use rig_agent::agent::{AgentBuilder, AgentHook, HookContext, ModelTurnAction, ModelTurnFinished};
use rig_agent::completion::{Cost, Usage as RigUsage};
use rig_agent::test_utils::{MockAddTool, MockCompletionModel, MockTurn};
use serde_json::json;

fn tool_turn(id: &str, usage: RigUsage) -> MockTurn {
    MockTurn::tool_call(id, "add", json!({"x": 1, "y": 1})).with_usage(usage)
}

fn budget(limit: f64) -> Arc<AgentBudget> {
    Arc::new(
        AgentBudget::builder("researcher", limit)
            .tier(ModelTier::Standard)
            .audit_log(AuditLog::Unbounded)
            .build()
            .unwrap(),
    )
}

async fn audit_sum(b: &AgentBudget) -> f64 {
    b.audit_log().await.iter().map(|r| r.cost_usd).sum()
}

/// gpt-4o: 10,000 prompt + 500 completion tokens = $0.025 + $0.005 = $0.03.
fn three_cent_turn() -> RigUsage {
    RigUsage::new().input_tokens(10_000).output_tokens(500)
}

#[tokio::test]
async fn the_turn_that_crosses_the_cap_is_booked_then_the_run_stops() {
    let b = budget(0.05);
    let model = MockCompletionModel::from_turns([
        tool_turn("a", three_cent_turn()),
        tool_turn("b", three_cent_turn()),
        tool_turn("c", three_cent_turn()),
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
    // Both paid turns are on the books (0.2.x dropped the second one).
    assert!((b.spent_usd() - 0.06).abs() < 1e-9, "spent {}", b.spent_usd());
    assert!((audit_sum(&b).await - b.spent_usd()).abs() < 1e-9);
    assert_eq!(b.audit_log().await.len(), 2);
    assert_eq!(b.reserved_usd(), 0.0);
    assert!(b.is_killed());
}

#[tokio::test]
async fn keep_alive_stops_the_run_without_killing_the_budget() {
    let b = budget(0.05);
    let model = MockCompletionModel::from_turns([
        tool_turn("a", three_cent_turn()),
        tool_turn("b", three_cent_turn()),
        MockTurn::text("done"),
    ]);
    let err = AgentBuilder::new(model)
        .tool(MockAddTool)
        .add_hook(BudgetHook::new(b.clone(), "gpt-4o").keep_alive_on_refusal())
        .build()
        .prompt("go")
        .max_turns(10)
        .await
        .unwrap_err();
    assert!(err.to_string().contains("budget exhausted"), "{err}");
    assert!(!b.is_killed());
    assert!((b.spent_usd() - 0.06).abs() < 1e-9);
}

#[tokio::test]
async fn a_killed_budget_stops_before_the_first_call() {
    let b = budget(1.0);
    b.kill("manual stop");
    let model = MockCompletionModel::from_turns([MockTurn::text("never sent")]);
    let err =
        AgentBuilder::new(model).add_hook(BudgetHook::new(b.clone(), "gpt-4o")).build().prompt("go").await.unwrap_err();
    assert!(err.to_string().contains("killed"), "{err}");
    assert_eq!(b.spent_usd(), 0.0);
}

#[tokio::test]
async fn a_call_whose_prompt_does_not_fit_is_never_made() {
    // $0.0001 buys 40 gpt-4o prompt tokens; this prompt is about 1,250.
    let b = budget(0.0001);
    let model = MockCompletionModel::from_turns([MockTurn::text("never sent").with_usage(three_cent_turn())]);
    let err = AgentBuilder::new(model)
        .add_hook(BudgetHook::new(b.clone(), "gpt-4o"))
        .build()
        .prompt("word ".repeat(1_000))
        .await
        .unwrap_err();
    assert!(err.to_string().contains("estimated"), "{err}");
    assert_eq!(b.spent_usd(), 0.0);
    assert_eq!(b.reserved_usd(), 0.0);
    assert!(b.audit_log().await.is_empty());
}

#[tokio::test]
async fn reserving_output_tokens_stops_earlier() {
    // $0.04 left after one $0.03 turn would admit another prompt-only
    // reservation, but not one that also holds 4,000 completion tokens
    // ($0.04) on top of the prompt.
    let b = budget(0.07);
    let model = MockCompletionModel::from_turns([
        tool_turn("a", three_cent_turn()),
        tool_turn("b", three_cent_turn()),
        MockTurn::text("done"),
    ]);
    let err = AgentBuilder::new(model)
        .tool(MockAddTool)
        .add_hook(BudgetHook::new(b.clone(), "gpt-4o").reserve_output_tokens(4_000))
        .build()
        .prompt("go")
        .max_turns(10)
        .await
        .unwrap_err();
    assert!(err.to_string().contains("estimated"), "{err}");
    assert!((b.spent_usd() - 0.03).abs() < 1e-9, "spent {}", b.spent_usd());
    assert!(b.spent_usd() <= b.limit_usd());
    assert_eq!(b.reserved_usd(), 0.0);
}

#[tokio::test]
async fn cache_reads_are_charged_at_the_cache_price() {
    let b = budget(1.0);
    let usage = RigUsage::new().input_tokens(2_100).output_tokens(50).cached_input_tokens(2_000);
    let model = MockCompletionModel::from_turns([MockTurn::text("ok").with_usage(usage)]);
    AgentBuilder::new(model).add_hook(BudgetHook::new(b.clone(), "gpt-4o")).build().prompt("go").await.unwrap();
    let expected = PriceTable::builtin().require("gpt-4o").unwrap().cost_usage(&Usage::from(&usage)).unwrap();
    let full = PriceTable::builtin().require("gpt-4o").unwrap().cost(2_100, 50);
    assert!((b.spent_usd() - expected).abs() < 1e-9);
    assert!(b.spent_usd() < full);
    let log = b.audit_log().await;
    assert_eq!(log[0].model.as_deref(), Some("gpt-4o"));
    assert_eq!((log[0].input_tokens, log[0].output_tokens), (2_100, 50));
}

#[tokio::test]
async fn a_provider_reported_cost_wins_over_the_table() {
    let b = budget(1.0);
    let usage = RigUsage::new().input_tokens(1_000).output_tokens(100).cost(Cost::from_parts(0.01, 0.002, 0.0, 0.0));
    let model = MockCompletionModel::from_turns([MockTurn::text("ok").with_usage(usage)]);
    AgentBuilder::new(model)
        .add_hook(BudgetHook::new(b.clone(), "some-model-not-in-the-table"))
        .build()
        .prompt("go")
        .await
        .unwrap();
    assert!((b.spent_usd() - 0.012).abs() < 1e-9);
}

#[tokio::test]
async fn runs_sharing_a_budget_book_every_paid_turn() {
    let b = budget(0.10);
    let mut answers = Vec::new();
    let mut stopped = 0;
    for _ in 0..5 {
        let model = MockCompletionModel::from_turns([MockTurn::text("ok").with_usage(three_cent_turn())]);
        let r = AgentBuilder::new(model).add_hook(BudgetHook::new(b.clone(), "gpt-4o")).build().prompt("go").await;
        match r {
            Ok(answer) => answers.push(answer),
            Err(_) => stopped += 1,
        }
    }
    // Runs 1-3 fit ($0.09). Run 4 is paid ($0.12): its answer is a final
    // one, so the caller gets it, and the kill switch trips. Run 5 never
    // calls the provider.
    assert_eq!(answers.len(), 4);
    assert_eq!(stopped, 1);
    assert!(b.is_killed());
    assert!((b.spent_usd() - 0.12).abs() < 1e-9, "spent {}", b.spent_usd());
    assert!((audit_sum(&b).await - b.spent_usd()).abs() < 1e-9);
}

#[tokio::test]
async fn an_over_cap_final_answer_is_returned_not_discarded() {
    // One $0.03 tool turn fits; the $0.03 final answer crosses the $0.05 cap.
    let b = budget(0.05);
    let model = MockCompletionModel::from_turns([
        tool_turn("a", three_cent_turn()),
        MockTurn::text("the answer").with_usage(three_cent_turn()),
    ]);
    let answer = AgentBuilder::new(model)
        .tool(MockAddTool)
        .add_hook(BudgetHook::new(b.clone(), "gpt-4o"))
        .build()
        .prompt("go")
        .max_turns(10)
        .await
        .unwrap();
    assert!(format!("{answer:?}").contains("the answer"), "{answer:?}");
    assert!((b.spent_usd() - 0.06).abs() < 1e-9, "spent {}", b.spent_usd());
    assert!(b.is_killed());
    assert_eq!(b.reserved_usd(), 0.0);
}

/// Stops every finished turn, like a token cap that has been reached.
struct StopEveryTurn;

impl AgentHook for StopEveryTurn {
    async fn on_model_turn_finished(&self, _ctx: &HookContext, _event: ModelTurnFinished<'_>) -> ModelTurnAction {
        ModelTurnAction::Stop("token cap reached".into())
    }
}

#[tokio::test]
async fn budget_hook_first_books_a_turn_a_later_hook_stops() {
    let b = budget(1.0);
    let model = MockCompletionModel::from_turns([tool_turn("a", three_cent_turn()), MockTurn::text("done")]);
    let err = AgentBuilder::new(model)
        .tool(MockAddTool)
        .add_hook(BudgetHook::new(b.clone(), "gpt-4o"))
        .add_hook(StopEveryTurn)
        .build()
        .prompt("go")
        .max_turns(10)
        .await
        .unwrap_err();
    assert!(err.to_string().contains("token cap reached"), "{err}");
    // The turn the second hook stopped on was paid for and is on the books.
    assert!((b.spent_usd() - 0.03).abs() < 1e-9, "spent {}", b.spent_usd());
    assert_eq!(b.audit_log().await.len(), 1);
    assert_eq!(b.reserved_usd(), 0.0);
}

/// Pins the documented limitation: rig stops dispatching a finished turn at
/// the first hook that does not return `Continue`, so a hook added before
/// `BudgetHook` that stops the turn keeps it off the books. If this starts
/// failing, rig changed its dispatch and the "Hook order" docs need updating.
#[tokio::test]
async fn a_hook_added_before_budget_hook_can_end_a_turn_unbooked() {
    let b = budget(1.0);
    let model = MockCompletionModel::from_turns([tool_turn("a", three_cent_turn()), MockTurn::text("done")]);
    let err = AgentBuilder::new(model)
        .tool(MockAddTool)
        .add_hook(StopEveryTurn)
        .add_hook(BudgetHook::new(b.clone(), "gpt-4o"))
        .build()
        .prompt("go")
        .max_turns(10)
        .await
        .unwrap_err();
    assert!(err.to_string().contains("token cap reached"), "{err}");
    assert_eq!(b.spent_usd(), 0.0);
    assert_eq!(b.reserved_usd(), 0.0);
}
