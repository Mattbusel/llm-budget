//! The README quickstart. Run with `cargo run --example quickstart`.

use llm_budget::{AgentBudget, BudgetError, FleetAllocator, ModelTier};

#[tokio::main]
async fn main() -> Result<(), BudgetError> {
    // One agent with a $5 cap. When 20% or less is left it is told to
    // move to a cheaper tier (Premium, Standard, Economy, Local).
    let agent = AgentBudget::new("researcher", 5.0, ModelTier::Premium, 0.2)?;

    // Charge a real call at the model's real price:
    // gpt-4o, 1,200 prompt tokens and 400 completion tokens.
    let receipt = agent.charge_model("gpt-4o", 1_200, 400, None, None).await?;
    println!("that call cost ${:.6}, ${:.4} left", receipt.cost_usd, agent.remaining_usd());

    // A fleet: $100 shared by several agents, each with its own cap.
    let fleet = FleetAllocator::new(100.0, ModelTier::Standard, 0.2)?;
    fleet.register_agent("planner", 40.0)?;
    fleet.register_agent("coder", 40.0)?;
    assert!(fleet.register_agent("extra", 40.0).is_err()); // would exceed $100

    fleet.charge_agent_model("coder", "claude-sonnet-4-5", 2_000, 800, None, None).await?;
    fleet.reallocate("planner", "coder", 10.0).await?; // move unspent budget

    // Emergency stop: every later charge fails with KillSwitchActivated.
    fleet.kill_all("runaway loop detected");
    assert!(fleet.charge_agent_model("coder", "gpt-4o", 10, 10, None, None).await.is_err());

    for snap in fleet.fleet_snapshot().await {
        println!("{} spent ${:.4} of ${:.2}", snap.agent_id, snap.spent_usd, snap.limit_usd);
    }
    Ok(())
}
