//! A budget that crosses process boundaries, offline: an orchestrator
//! delegates to a remote A2A agent, which calls an MCP tool server.
//!
//! ```text
//! cargo run --example budget_context
//! ```
//!
//! 1. The orchestrator reserves a $0.30 slice and sends it in A2A message
//!    metadata (the budget/v1 extension draft, a2aproject/A2A#2121).
//! 2. The remote agent builds its budget from that, spends some, and passes
//!    what it has left to an MCP tool server in `params._meta.baggage`
//!    (W3C baggage, which MCP reserves in `_meta` per SEP-414).
//! 3. The tool server reads the cap from the baggage; the agent books the
//!    tool's cost and reports `incurred` back. The orchestrator books the
//!    report on the slice once, even if the final task is resent.

use llm_budget::context::{a2a, mcp, BudgetContext, IncurredBooker};
use llm_budget::{AgentBudget, BudgetError, ChargeContext, Limit};
use serde_json::json;

fn main() -> Result<(), BudgetError> {
    // ── orchestrator ──
    let orchestrator = AgentBudget::builder("orchestrator", 1.00).build()?;
    let slice = orchestrator.child("remote-researcher", Limit::Usd(0.30))?;
    let outbound = BudgetContext::outbound(&slice).with_deadline("2026-12-01T00:00:00Z");
    let request = json!({
        "message": {"role": "ROLE_USER", "parts": [{"text": "research X"}]},
        "metadata": a2a::to_metadata(&outbound)?,
    });
    println!("A2A request metadata:\n{}", serde_json::to_string_pretty(&request["metadata"]).unwrap_or_default());

    // ── remote agent ──
    let received = a2a::from_metadata(&request["metadata"])?
        .ok_or_else(|| BudgetError::InvalidConfig("no budget in the request".into()))?;
    let agent = AgentBudget::from_context("researcher", &received)?;
    agent.reserve(0.05)?.settle(0.04); // its own model call

    let tool_slice = agent.child("search-tool", Limit::Usd(0.10))?;
    let mut meta = serde_json::Map::new();
    meta.insert("traceparent".into(), json!("00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"));
    mcp::inject(&mut meta, &BudgetContext::outbound(&tool_slice));
    let tool_call = json!({"jsonrpc": "2.0", "id": 7, "method": "tools/call",
        "params": {"name": "search", "arguments": {"q": "X"}, "_meta": meta}});
    println!("\nMCP tools/call _meta.baggage:\n{}", tool_call["params"]["_meta"]["baggage"]);

    // ── MCP tool server ──
    let tool_budget = mcp::extract(&tool_call["params"]["_meta"])?
        .and_then(|c| c.budget_usd())
        .ok_or_else(|| BudgetError::InvalidConfig("no budget in _meta".into()))?;
    println!("tool server may spend up to ${}", tool_budget);
    let tool_cost = 0.02; // what the tool reports it spent

    // ── remote agent books the tool and reports back ──
    tool_slice.charge_usd_force(tool_cost, ChargeContext::new().task("search"));
    drop(tool_slice);
    let report = BudgetContext::reporting(&agent);
    let task_metadata = a2a::to_metadata(&report)?;
    println!(
        "\nA2A task metadata on completion:\n{}",
        serde_json::to_string_pretty(&task_metadata).unwrap_or_default()
    );

    // ── orchestrator books the report (twice received, booked once) ──
    let mut booker = IncurredBooker::new();
    for _ in 0..2 {
        let incurred = a2a::from_metadata(&task_metadata)?.and_then(|c| c.incurred);
        booker.book(&slice, incurred.as_ref());
    }
    let receipt = slice.settle();
    println!(
        "\norchestrator: remote agent cost ${:.2}, ${:.2} of its slice refunded, ${:.2} spent of ${:.2}",
        receipt.spent_usd,
        receipt.released_usd,
        orchestrator.spent_usd(),
        orchestrator.limit_usd()
    );
    Ok(())
}
