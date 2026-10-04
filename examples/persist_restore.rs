//! Save a fleet's spend to disk at the end of a run and pick up where you left
//! off next time, so a restart does not reset anyone's budget.
//!
//! Run with `cargo run --example persist_restore`.

use llm_budget::{AgentBudget, BudgetError, BudgetStore, FleetAllocator, ModelTier};

#[tokio::main]
async fn main() -> Result<(), BudgetError> {
    let dir = std::env::temp_dir().join("llm-budget-example");
    let store = BudgetStore::new(&dir);

    // Run 1: two agents do some work.
    let fleet = FleetAllocator::new(20.0, ModelTier::Standard, 0.2)?;
    fleet.register_agent("researcher", 10.0)?;
    fleet.register_agent("writer", 10.0)?;
    fleet.charge_agent_model("researcher", "gpt-4o", 400_000, 20_000, None, None).await?;
    fleet.charge_agent_model("writer", "claude-sonnet-4-5", 50_000, 30_000, None, None).await?;
    store.save("nightly-run", fleet.fleet_snapshot().await).await?;
    println!("saved to {}", dir.display());

    // Run 2 (a new process in real life): rebuild each agent with only what is left.
    let saved = BudgetStore::new(&dir).load("nightly-run").await?;
    for snap in &saved.snapshots {
        let agent = AgentBudget::new(snap.agent_id.clone(), snap.remaining_usd, snap.model_tier, 0.2)?;
        println!(
            "{:<10} spent ${:.4} last run, ${:.4} left for this one",
            agent.agent_id,
            snap.spent_usd,
            agent.remaining_usd()
        );
    }

    store.delete("nightly-run").await?;
    Ok(())
}
