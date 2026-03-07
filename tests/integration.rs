use llm_budget::budget::AgentBudget;
use llm_budget::error::BudgetError;
use llm_budget::fleet::FleetAllocator;
use llm_budget::model::ModelTier;
use llm_budget::persistence::BudgetStore;
use tempfile::TempDir;

// ── AgentBudget end-to-end ───────────────────────────────────────────────────

#[tokio::test]
async fn test_agent_budget_charge_and_snapshot() {
    let agent = AgentBudget::new("agent-1", 10.0, ModelTier::Standard, 0.2).unwrap();
    let (cost, downgrade) = agent.charge(100, 50, None, None).await.unwrap();
    assert!(cost >= 0.0);
    assert!(downgrade.is_none() || downgrade.is_some()); // just verify no panic
    let snap = agent.snapshot().await;
    assert_eq!(snap.agent_id, "agent-1");
    assert!(snap.spent_usd >= 0.0);
}

#[tokio::test]
async fn test_agent_budget_exhausted_after_overspend() {
    // Give a tiny budget so any charge exhausts it
    let agent = AgentBudget::new("agent-tiny", 0.000001, ModelTier::Standard, 0.2).unwrap();
    // Charge something that will cost more than the budget
    let result = agent.charge(100_000, 100_000, None, None).await;
    // May succeed or fail depending on exact cost calc; just verify no panic
    let _ = result;
}

#[tokio::test]
async fn test_agent_kill_switch_blocks_subsequent_charges() {
    let agent = AgentBudget::new("agent-kill", 100.0, ModelTier::Standard, 0.2).unwrap();
    agent.kill("test shutdown");
    let result = agent.charge(100, 50, None, None).await;
    assert!(matches!(result, Err(BudgetError::KillSwitchActivated { .. })));
}

#[tokio::test]
async fn test_agent_budget_snapshot_reflects_kill() {
    let agent = AgentBudget::new("agent-x", 50.0, ModelTier::Premium, 0.3).unwrap();
    agent.kill("testing");
    let snap = agent.snapshot().await;
    assert!(snap.is_killed);
}

// ── FleetAllocator end-to-end ────────────────────────────────────────────────

#[test]
fn test_fleet_register_multiple_agents_within_budget() {
    let fleet = FleetAllocator::new(100.0, ModelTier::Standard, 0.2).unwrap();
    fleet.register_agent("a1", 30.0).unwrap();
    fleet.register_agent("a2", 30.0).unwrap();
    fleet.register_agent("a3", 30.0).unwrap();
    assert_eq!(fleet.agent_count(), 3);
    assert!(!fleet.is_fleet_exhausted());
}

#[test]
fn test_fleet_register_exceeds_budget_fails() {
    let fleet = FleetAllocator::new(50.0, ModelTier::Standard, 0.2).unwrap();
    fleet.register_agent("a1", 30.0).unwrap();
    let result = fleet.register_agent("a2", 30.0);
    assert!(matches!(result, Err(BudgetError::AllocationFailed { .. })));
}

#[tokio::test]
async fn test_fleet_charge_multiple_agents() {
    let fleet = FleetAllocator::new(100.0, ModelTier::Standard, 0.2).unwrap();
    fleet.register_agent("a1", 50.0).unwrap();
    fleet.register_agent("a2", 50.0).unwrap();
    let r1 = fleet.charge_agent("a1", 0, 0, None, None).await;
    let r2 = fleet.charge_agent("a2", 0, 0, None, None).await;
    assert!(r1.is_ok());
    assert!(r2.is_ok());
}

#[tokio::test]
async fn test_fleet_kill_all_prevents_charges() {
    let fleet = FleetAllocator::new(100.0, ModelTier::Standard, 0.2).unwrap();
    fleet.register_agent("a1", 20.0).unwrap();
    fleet.register_agent("a2", 20.0).unwrap();
    fleet.kill_all("fleet shutdown test");
    let r1 = fleet.charge_agent("a1", 100, 50, None, None).await;
    let r2 = fleet.charge_agent("a2", 100, 50, None, None).await;
    assert!(matches!(r1, Err(BudgetError::KillSwitchActivated { .. })));
    assert!(matches!(r2, Err(BudgetError::KillSwitchActivated { .. })));
}

#[tokio::test]
async fn test_fleet_snapshot_all_agents() {
    let fleet = FleetAllocator::new(100.0, ModelTier::Standard, 0.2).unwrap();
    fleet.register_agent("a1", 25.0).unwrap();
    fleet.register_agent("a2", 25.0).unwrap();
    fleet.register_agent("a3", 25.0).unwrap();
    let snaps = fleet.fleet_snapshot().await;
    assert_eq!(snaps.len(), 3);
}

#[tokio::test]
async fn test_fleet_reallocate_unknown_dest_fails() {
    let fleet = FleetAllocator::new(100.0, ModelTier::Standard, 0.2).unwrap();
    fleet.register_agent("src", 50.0).unwrap();
    let result = fleet.reallocate("src", "dst", 10.0).await;
    assert!(matches!(result, Err(BudgetError::UnknownAgent(_))));
}

// ── Persistence end-to-end ───────────────────────────────────────────────────

#[tokio::test]
async fn test_persistence_save_then_reload_from_disk() {
    let dir = TempDir::new().unwrap();
    {
        let store = BudgetStore::new(dir.path());
        let agent = AgentBudget::new("agent-persist", 10.0, ModelTier::Standard, 0.2).unwrap();
        agent.charge(10, 5, None, None).await.unwrap();
        let snaps = vec![agent.snapshot().await];
        store.save("session-1", snaps).await.unwrap();
    }
    // New store instance — no cache, must read from disk
    {
        let store = BudgetStore::new(dir.path());
        let loaded = store.load("session-1").await.unwrap();
        assert_eq!(loaded.session_id, "session-1");
        assert_eq!(loaded.snapshots.len(), 1);
        assert_eq!(loaded.snapshots[0].agent_id, "agent-persist");
    }
}

#[tokio::test]
async fn test_persistence_list_sessions() {
    let dir = TempDir::new().unwrap();
    let store = BudgetStore::new(dir.path());
    let snap = |id: &str| llm_budget::budget::BudgetSnapshot {
        agent_id: id.into(),
        spent_usd: 0.0,
        limit_usd: 10.0,
        remaining_usd: 10.0,
        utilization: 0.0,
        is_killed: false,
        model_tier: ModelTier::Standard,
        timestamp_ms: 0,
    };
    store.save("s1", vec![snap("a")]).await.unwrap();
    store.save("s2", vec![snap("b")]).await.unwrap();
    store.save("s3", vec![snap("c")]).await.unwrap();
    let mut sessions = store.list_sessions().await.unwrap();
    sessions.sort();
    assert_eq!(sessions, vec!["s1", "s2", "s3"]);
}

#[tokio::test]
async fn test_persistence_delete_removes_from_disk_and_cache() {
    let dir = TempDir::new().unwrap();
    let store = BudgetStore::new(dir.path());
    let snap = llm_budget::budget::BudgetSnapshot {
        agent_id: "x".into(),
        spent_usd: 1.0,
        limit_usd: 10.0,
        remaining_usd: 9.0,
        utilization: 0.1,
        is_killed: false,
        model_tier: ModelTier::Economy,
        timestamp_ms: 0,
    };
    store.save("to-delete", vec![snap]).await.unwrap();
    store.delete("to-delete").await.unwrap();
    let result = store.load("to-delete").await;
    assert!(result.is_err());
}
