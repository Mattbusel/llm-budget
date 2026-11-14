//! A dollar cap on a rig agent, run offline against rig's mock model.
//!
//! ```text
//! cargo run --example rig_budget_hook --features __rig-test-utils
//! ```
//!
//! With a real provider the only change is the model you hand to
//! `AgentBuilder::new`: the hook, the budget and the numbers work the same.
//!
//! What it shows:
//! 1. Each call is authorized before it is sent (a reservation sized from the
//!    prompt plus `reserve_output_tokens`) and booked after it returns, at
//!    gpt-4o's real price.
//! 2. The turn that crosses the cap is still booked, because it was paid
//!    for, and only then does the run stop.
//! 3. Two agents can share one `Arc<AgentBudget>`; once the cap is reached
//!    the kill switch stops the second one before it pays for anything.

use std::sync::Arc;

use llm_budget::rig::BudgetHook;
use llm_budget::{AgentBudget, AuditLog, BudgetError, ModelTier};
use rig_agent::agent::AgentBuilder;
use rig_agent::completion::Usage;
use rig_agent::test_utils::{MockAddTool, MockCompletionModel, MockTurn};

/// A tool-calling turn that bills 10,000 prompt + 500 completion tokens:
/// $0.03 on gpt-4o.
fn paid_turn(id: &str) -> MockTurn {
    MockTurn::tool_call(id, "add", serde_json::json!({"x": 1, "y": 2}))
        .with_usage(Usage::new().input_tokens(10_000).output_tokens(500))
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), BudgetError> {
    // $0.10 for the whole team, every booked turn kept in the audit log.
    let budget = Arc::new(
        AgentBudget::builder("research-team", 0.10)
            .tier(ModelTier::Standard)
            .audit_log(AuditLog::Unbounded)
            .build()?,
    );

    // Agent 1 wants five paid turns, but the budget covers three and a bit.
    let researcher = AgentBuilder::new(MockCompletionModel::from_turns([
        paid_turn("a"),
        paid_turn("b"),
        paid_turn("c"),
        paid_turn("d"),
        paid_turn("e"),
        MockTurn::text("done"),
    ]))
    .tool(MockAddTool)
    .add_hook(BudgetHook::new(budget.clone(), "gpt-4o").reserve_output_tokens(500))
    .build();

    match researcher.prompt("research the topic").max_turns(10).await {
        Ok(answer) => println!("researcher finished: {answer:?}"),
        Err(e) => println!("researcher stopped: {e}"),
    }
    println!(
        "spent ${:.4} of ${:.2} (killed: {})",
        budget.spent_usd(),
        budget.limit_usd(),
        budget.is_killed()
    );

    // Agent 2 shares the budget. It is stopped before its first call, so
    // it never costs anything.
    let writer = AgentBuilder::new(MockCompletionModel::from_turns([MockTurn::text("draft")]))
        .add_hook(BudgetHook::new(budget.clone(), "gpt-4o"))
        .build();
    if let Err(e) = writer.prompt("write it up").await {
        println!("writer stopped before calling the model: {e}");
    }

    // Every paid turn is on the books, and the log adds up to the total.
    let log = budget.audit_log().await;
    for r in &log {
        println!(
            "  {} {} in / {} out  ${:.4}",
            r.model.as_deref().unwrap_or("?"),
            r.input_tokens,
            r.output_tokens,
            r.cost_usd
        );
    }
    let sum: f64 = log.iter().map(|r| r.cost_usd).sum();
    assert!((sum - budget.spent_usd()).abs() < 1e-9);
    assert_eq!(budget.reserved_usd(), 0.0);
    Ok(())
}
