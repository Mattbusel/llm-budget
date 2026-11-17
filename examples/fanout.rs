//! Four sub-agents in parallel under one $1.00 cap, offline.
//!
//! ```text
//! cargo run --example fanout
//! ```
//!
//! Each sub-agent wants eight gpt-4o calls. A call is authorized with a
//! reservation sized for a 20,000-token prompt and 1,000 output tokens
//! ($0.06 at gpt-4o's price) and settled at what the canned "provider"
//! reports (18,000 in, 700 out: $0.052).
//!
//! 1. **Passing the remainder down.** Each sub-agent is told how much the
//!    parent has left and enforces that on its own. All four start at the
//!    same moment, so all four see $1.00, and together they spend $1.664.
//! 2. **Child scopes.** Each sub-agent gets `parent.child(.., Limit::Usd(0.25))`:
//!    its slice is reserved from the parent before it starts, its calls are
//!    booked on the parent as they settle, and the unused part of the slice
//!    goes back when it finishes. Total spend stays under $1.00.

use std::sync::Arc;

use llm_budget::{AgentBudget, BudgetError, Limit, ScopeGuard, Usage};

const MODEL: &str = "gpt-4o";
const CALLS: usize = 8;

/// One sub-agent's loop: up to eight calls, each reserved before it is
/// "sent" and settled with the usage the provider reported.
async fn sub_agent(budget: &AgentBudget) -> usize {
    let mut made = 0;
    for _ in 0..CALLS {
        let hold = match budget.reserve_for(MODEL, &Usage::new(20_000, 1_000)) {
            Ok(h) => h,
            Err(_) => break, // refused before the call: nothing spent
        };
        tokio::task::yield_now().await; // the model call would go here
        hold.settle_usage(MODEL, &Usage::new(18_000, 700));
        made += 1;
    }
    made
}

#[tokio::main]
async fn main() -> Result<(), BudgetError> {
    // 1. Every sub-agent gets its own budget equal to the parent's remainder.
    let parent = Arc::new(AgentBudget::builder("orchestrator", 1.00).build()?);
    let remaining = parent.remaining_usd();
    let mut tasks = Vec::new();
    for i in 0..4 {
        let parent = parent.clone();
        tasks.push(tokio::spawn(async move {
            let own = AgentBudget::builder(format!("sub-{i}"), remaining).build()?;
            let calls = sub_agent(&own).await;
            parent.charge_usd_force(own.spent_usd(), Default::default());
            Ok::<usize, BudgetError>(calls)
        }));
    }
    let mut calls = 0;
    for t in tasks {
        calls += t.await.map_err(|e| BudgetError::InvalidConfig(e.to_string()))??;
    }
    println!(
        "remainder passed down: {calls} calls, spent ${:.3} of ${:.2} ({:.0}% over the cap)",
        parent.spent_usd(),
        parent.limit_usd(),
        (parent.spent_usd() / parent.limit_usd() - 1.0) * 100.0
    );

    // 2. Child scopes: each slice is reserved from the parent up front.
    let parent = Arc::new(AgentBudget::builder("orchestrator", 1.00).build()?);
    let mut tasks = Vec::new();
    for i in 0..4 {
        let scope: ScopeGuard = parent.child(format!("sub-{i}"), Limit::Usd(0.25))?;
        tasks.push(tokio::spawn(async move {
            let calls = sub_agent(&scope).await;
            (calls, scope.settle())
        }));
    }
    let mut calls = 0;
    for t in tasks {
        let (n, receipt) = t.await.map_err(|e| BudgetError::InvalidConfig(e.to_string()))?;
        calls += n;
        println!(
            "  {}: {n} calls, spent ${:.3} of its ${:.2}, ${:.3} back to the parent",
            receipt.scope_id, receipt.spent_usd, receipt.held_usd, receipt.released_usd
        );
    }
    println!(
        "child scopes:          {calls} calls, spent ${:.3} of ${:.2}, ${:.3} still held",
        parent.spent_usd(),
        parent.limit_usd(),
        parent.reserved_usd()
    );
    assert!(parent.spent_usd() <= parent.limit_usd());
    Ok(())
}
