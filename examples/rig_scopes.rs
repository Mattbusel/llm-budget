//! A rig orchestrator that delegates to two sub-agents, each in its own child
//! scope of one $0.10 budget. Offline, against rig's mock model.
//!
//! ```text
//! cargo run --example rig_scopes --features __rig-test-utils
//! ```
//!
//! Each sub-agent gets `BudgetHook::new(scope.budget().clone(), ..)`, so its
//! turns are reserved and booked on its own slice, and on the parent at the
//! same moment. The researcher's slice runs out first and only the
//! researcher stops; the writer keeps its own slice. When both are done the
//! unused part of each slice goes back to the parent.

use std::sync::Arc;

use llm_budget::rig::BudgetHook;
use llm_budget::{AgentBudget, AuditLog, BudgetError, Limit};
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
    let parent = Arc::new(AgentBudget::builder("orchestrator", 0.10).audit_log(AuditLog::Unbounded).build()?);

    // $0.06 for research, half of what is left ($0.02) for writing.
    let research = parent.child("researcher", Limit::Usd(0.06))?;
    let writing = parent.child("writer", Limit::Fraction(0.5))?;
    println!(
        "slices: researcher ${:.2}, writer ${:.2}; orchestrator has ${:.2} unallocated",
        research.limit_usd(),
        writing.limit_usd(),
        parent.remaining_usd()
    );

    // The researcher wants four paid turns; its slice covers two.
    let researcher = AgentBuilder::new(MockCompletionModel::from_turns([
        paid_turn("a"),
        paid_turn("b"),
        paid_turn("c"),
        paid_turn("d"),
        MockTurn::text("findings"),
    ]))
    .tool(MockAddTool)
    .add_hook(BudgetHook::new(research.budget().clone(), "gpt-4o").reserve_output_tokens(500))
    .build();
    match researcher.prompt("research the topic").max_turns(10).await {
        Ok(_) => println!("researcher finished"),
        Err(e) => println!("researcher stopped: {e}"),
    }

    // The hook killed the researcher's scope only; the writer's slice is intact.
    let writer = AgentBuilder::new(MockCompletionModel::from_turns([
        MockTurn::text("draft").with_usage(Usage::new().input_tokens(2_000).output_tokens(300)),
    ]))
    .add_hook(BudgetHook::new(writing.budget().clone(), "gpt-4o").reserve_output_tokens(300))
    .build();
    match writer.prompt("write it up").await {
        Ok(_) => println!("writer finished its draft"),
        Err(e) => println!("writer stopped: {e}"),
    }

    for receipt in [research.settle(), writing.settle()] {
        println!(
            "{}: spent ${:.4} of ${:.2}, ${:.4} back to the orchestrator",
            receipt.scope_id, receipt.spent_usd, receipt.held_usd, receipt.released_usd
        );
    }
    println!(
        "orchestrator: spent ${:.4} of ${:.2} (all of it by sub-agents: ${:.4}), ${:.4} left",
        parent.spent_usd(),
        parent.limit_usd(),
        parent.child_cost_usd(),
        parent.remaining_usd()
    );
    // Every sub-agent turn is in the orchestrator's audit log, under the sub-agent's name.
    for r in parent.audit_log().await {
        println!("  {:<10} ${:.4}", r.agent_id, r.cost_usd);
    }
    Ok(())
}
