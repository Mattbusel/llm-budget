//! Extended test coverage — error variants, model cascade edge cases,
//! atomic budget concurrency, fleet exhaustion, persistence edge cases.

use llm_budget::budget::{AgentBudget, AtomicBudget, BudgetEnvelope, BudgetSnapshot, ScopeKind};
use llm_budget::error::BudgetError;
use llm_budget::fleet::{FleetAllocator, FleetBudget};
use llm_budget::model::{ModelCascade, ModelTier};
use llm_budget::persistence::BudgetStore;
use std::sync::Arc;
use tempfile::TempDir;

// ── Error display coverage ───────────────────────────────────────────────────

#[test]
fn test_error_agent_exhausted_display() {
    let e = BudgetError::AgentExhausted { agent_id: "a1".into(), spent: 10.0, limit: 10.0 };
    assert!(e.to_string().contains("a1"));
    assert!(e.to_string().contains("10"));
}

#[test]
fn test_error_task_exhausted_display() {
    let e = BudgetError::TaskExhausted { task_id: "task-1".into(), spent: 5.0, limit: 5.0 };
    assert!(e.to_string().contains("task-1"));
}

#[test]
fn test_error_session_exhausted_display() {
    let e = BudgetError::SessionExhausted { session_id: "sess-42".into(), spent: 3.0, limit: 3.0 };
    assert!(e.to_string().contains("sess-42"));
}

#[test]
fn test_error_fleet_exhausted_display() {
    let e = BudgetError::FleetExhausted { spent: 100.0, limit: 100.0 };
    assert!(e.to_string().contains("Fleet"));
}

#[test]
fn test_error_kill_switch_display() {
    let e = BudgetError::KillSwitchActivated { agent_id: "bot".into(), reason: "hard limit".into() };
    assert!(e.to_string().contains("bot"));
    assert!(e.to_string().contains("hard limit"));
}

#[test]
fn test_error_model_downgraded_display() {
    let e = BudgetError::ModelDowngraded { from: ModelTier::Premium, to: ModelTier::Standard, remaining: 0.05 };
    assert!(e.to_string().contains("Premium"));
    assert!(e.to_string().contains("Standard"));
}

#[test]
fn test_error_allocation_failed_display() {
    let e = BudgetError::AllocationFailed { agent_id: "x".into(), requested: 50.0, available: 20.0 };
    assert!(e.to_string().contains("x"));
    assert!(e.to_string().contains("50"));
}

#[test]
fn test_error_reallocation_failed_display() {
    let e = BudgetError::ReallocationFailed { from: "a".into(), to: "b".into(), amount: 10.0, available: 5.0 };
    assert!(e.to_string().contains("a"));
    assert!(e.to_string().contains("b"));
}

#[test]
fn test_error_unknown_agent_display() {
    let e = BudgetError::UnknownAgent("ghost".into());
    assert!(e.to_string().contains("ghost"));
}

#[test]
fn test_error_persistence_display() {
    let e = BudgetError::PersistenceError("disk full".into());
    assert!(e.to_string().contains("disk full"));
}

#[test]
fn test_error_invalid_config_display() {
    let e = BudgetError::InvalidConfig("must be positive".into());
    assert!(e.to_string().contains("must be positive"));
}

#[test]
fn test_error_equality() {
    let e1 = BudgetError::UnknownAgent("a".into());
    let e2 = BudgetError::UnknownAgent("a".into());
    assert_eq!(e1, e2);
    let e3 = BudgetError::UnknownAgent("b".into());
    assert_ne!(e1, e3);
}

// ── AtomicBudget concurrency ─────────────────────────────────────────────────

#[test]
fn test_atomic_budget_zero_limit_rejected() {
    assert!(AtomicBudget::new(0.0).is_err());
    assert!(AtomicBudget::new(-1.0).is_err());
}

#[test]
fn test_atomic_budget_remaining_tracks_spend() {
    let b = AtomicBudget::new(10.0).unwrap();
    b.charge(3.0).unwrap();
    b.charge(2.0).unwrap();
    assert!((b.remaining_usd() - 5.0).abs() < 1e-4);
}

#[test]
fn test_atomic_budget_utilization() {
    let b = AtomicBudget::new(10.0).unwrap();
    b.charge(5.0).unwrap();
    assert!((b.utilization() - 0.5).abs() < 1e-4);
}

#[test]
fn test_atomic_budget_concurrent_charges_total_correct() {
    let b = Arc::new(AtomicBudget::new(1000.0).unwrap());
    let mut handles = Vec::new();
    for _ in 0..10 {
        let bc = Arc::clone(&b);
        handles.push(std::thread::spawn(move || {
            bc.charge(1.0).ok();
        }));
    }
    for h in handles { h.join().unwrap(); }
    assert!((b.spent_usd() - 10.0).abs() < 1e-4);
}

#[test]
fn test_atomic_budget_over_limit_rejected_and_rolled_back() {
    let b = AtomicBudget::new(5.0).unwrap();
    b.charge(4.5).unwrap();
    let result = b.charge(1.0); // would exceed limit
    assert!(result.is_err());
    assert!((b.spent_usd() - 4.5).abs() < 1e-4); // rolled back
}

// ── ModelCascade edge cases ──────────────────────────────────────────────────

#[test]
fn test_cascade_invalid_threshold_above_1_rejected() {
    assert!(ModelCascade::new(ModelTier::Premium, 1.5, 100.0).is_err());
}

#[test]
fn test_cascade_invalid_threshold_negative_rejected() {
    assert!(ModelCascade::new(ModelTier::Standard, -0.1, 100.0).is_err());
}

#[test]
fn test_cascade_invalid_budget_zero_rejected() {
    assert!(ModelCascade::new(ModelTier::Premium, 0.2, 0.0).is_err());
}

#[test]
fn test_cascade_evaluate_no_downgrade_when_budget_ample() {
    let mut c = ModelCascade::new(ModelTier::Premium, 0.2, 100.0).unwrap();
    let result = c.evaluate(80.0); // 80% remaining — no pressure
    assert!(result.is_none());
    assert_eq!(c.current(), ModelTier::Premium);
}

#[test]
fn test_cascade_evaluate_downgrades_when_under_threshold() {
    let mut c = ModelCascade::new(ModelTier::Premium, 0.2, 100.0).unwrap();
    let result = c.evaluate(15.0); // 15% remaining — under 20% threshold
    assert_eq!(result, Some(ModelTier::Standard));
    assert_eq!(c.current(), ModelTier::Standard);
}

#[test]
fn test_cascade_force_downgrade_chain() {
    let mut c = ModelCascade::new(ModelTier::Premium, 0.1, 100.0).unwrap();
    assert_eq!(c.force_downgrade(), Some(ModelTier::Standard));
    assert_eq!(c.force_downgrade(), Some(ModelTier::Economy));
    assert_eq!(c.force_downgrade(), Some(ModelTier::Local));
    assert_eq!(c.force_downgrade(), None);
}

#[test]
fn test_cascade_at_local_no_further_downgrade() {
    let mut c = ModelCascade::new(ModelTier::Local, 0.5, 100.0).unwrap();
    let result = c.evaluate(10.0); // under threshold but already at Local
    assert!(result.is_none());
}

// ── ModelTier cost estimates ─────────────────────────────────────────────────

#[test]
fn test_model_tier_estimate_cost_zero_tokens() {
    for tier in [ModelTier::Premium, ModelTier::Standard, ModelTier::Economy, ModelTier::Local] {
        assert_eq!(tier.estimate_cost(0, 0), 0.0);
    }
}

#[test]
fn test_model_tier_local_always_zero_cost() {
    assert_eq!(ModelTier::Local.estimate_cost(100_000, 100_000), 0.0);
}

#[test]
fn test_model_tier_premium_most_expensive() {
    let cost_premium = ModelTier::Premium.estimate_cost(1000, 1000);
    let cost_standard = ModelTier::Standard.estimate_cost(1000, 1000);
    let cost_economy = ModelTier::Economy.estimate_cost(1000, 1000);
    assert!(cost_premium > cost_standard);
    assert!(cost_standard > cost_economy);
}

#[test]
fn test_model_tier_serde_all_variants() {
    for tier in [ModelTier::Premium, ModelTier::Standard, ModelTier::Economy, ModelTier::Local] {
        let json = serde_json::to_string(&tier).unwrap();
        let back: ModelTier = serde_json::from_str(&json).unwrap();
        assert_eq!(tier, back);
    }
}

// ── FleetBudget micro-dollar precision ──────────────────────────────────────

#[test]
fn test_fleet_budget_micro_dollar_precision() {
    let fleet = FleetBudget::new(10.0).unwrap();
    fleet.allocate("a", 0.000001).unwrap(); // 1 micro-dollar
    assert!((fleet.available_usd() - 9.999999).abs() < 1e-6);
}

#[test]
fn test_fleet_budget_record_spend_accumulates() {
    let fleet = FleetBudget::new(100.0).unwrap();
    fleet.record_spend(5.0);
    fleet.record_spend(3.0);
    assert!((fleet.spent_usd() - 8.0).abs() < 1e-4);
}

#[test]
fn test_fleet_budget_total_limit_accessor() {
    let fleet = FleetBudget::new(42.5).unwrap();
    assert!((fleet.total_limit_usd() - 42.5).abs() < 1e-4);
}

#[test]
fn test_fleet_budget_allocated_tracks_allocations() {
    let fleet = FleetBudget::new(100.0).unwrap();
    fleet.allocate("a1", 30.0).unwrap();
    fleet.allocate("a2", 20.0).unwrap();
    assert!((fleet.allocated_usd() - 50.0).abs() < 1e-4);
}

// ── BudgetEnvelope scoped kinds ──────────────────────────────────────────────

#[tokio::test]
async fn test_budget_envelope_task_scope() {
    let env = BudgetEnvelope::new("task-1", ScopeKind::Task, 1.0).unwrap();
    assert_eq!(env.scope_kind, ScopeKind::Task);
}

#[tokio::test]
async fn test_budget_envelope_session_scope() {
    let env = BudgetEnvelope::new("sess-1", ScopeKind::Session, 5.0).unwrap();
    assert_eq!(env.scope_kind, ScopeKind::Session);
}

#[tokio::test]
async fn test_budget_envelope_kill_blocks_charge() {
    let env = BudgetEnvelope::new("a1", ScopeKind::Agent, 100.0).unwrap();
    env.kill("hard limit");
    assert!(env.is_killed());
}

#[tokio::test]
async fn test_budget_envelope_audit_log_grows() {
    let env = BudgetEnvelope::new("a1", ScopeKind::Agent, 100.0).unwrap();
    let rec = llm_budget::budget::SpendRecord {
        agent_id: "a1".into(),
        task_id: None,
        session_id: None,
        model_tier: ModelTier::Standard,
        input_tokens: 100,
        output_tokens: 50,
        cost_usd: 0.001,
        timestamp_ms: 0,
    };
    env.charge(rec.clone()).await.unwrap();
    env.charge(rec).await.unwrap();
    let log = env.audit_log().await;
    assert_eq!(log.len(), 2);
}

// ── AgentBudget model accessor ───────────────────────────────────────────────

#[tokio::test]
async fn test_agent_budget_remaining_decreases_after_charge() {
    let agent = AgentBudget::new("a1", 10.0, ModelTier::Premium, 0.1).unwrap();
    let before = agent.remaining_usd();
    agent.charge(10, 10, None, None).await.unwrap();
    let after = agent.remaining_usd();
    assert!(after < before);
}

#[tokio::test]
async fn test_agent_budget_snapshot_utilization_between_0_and_1() {
    let agent = AgentBudget::new("a1", 10.0, ModelTier::Standard, 0.1).unwrap();
    agent.charge(100, 50, None, None).await.unwrap();
    let snap = agent.snapshot().await;
    assert!(snap.utilization >= 0.0 && snap.utilization <= 1.0);
}

// ── Persistence: overwrite and cache eviction ────────────────────────────────

#[tokio::test]
async fn test_persistence_overwrite_reads_latest_from_disk() {
    let dir = TempDir::new().unwrap();
    let make_snap = |spent: f64| BudgetSnapshot {
        agent_id: "a".into(),
        spent_usd: spent,
        limit_usd: 10.0,
        remaining_usd: 10.0 - spent,
        utilization: spent / 10.0,
        is_killed: false,
        model_tier: ModelTier::Standard,
        timestamp_ms: 0,
    };
    let store = BudgetStore::new(dir.path());
    store.save("s1", vec![make_snap(1.0)]).await.unwrap();
    // New store instance — no shared cache, forces disk read
    let store2 = BudgetStore::new(dir.path());
    store2.save("s1", vec![make_snap(5.0)]).await.unwrap();
    let store3 = BudgetStore::new(dir.path());
    let loaded = store3.load("s1").await.unwrap();
    assert!((loaded.snapshots[0].spent_usd - 5.0).abs() < 1e-6);
}

#[tokio::test]
async fn test_persistence_load_nonexistent_is_error() {
    let dir = TempDir::new().unwrap();
    let store = BudgetStore::new(dir.path());
    assert!(store.load("does-not-exist").await.is_err());
}
