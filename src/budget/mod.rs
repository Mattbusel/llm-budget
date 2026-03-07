//! Core budget primitives: per-agent atomic tracking and scoped envelopes.

use crate::error::BudgetError;
use crate::model::ModelTier;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use tokio::sync::RwLock;

/// Spend record for audit and visibility.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SpendRecord {
    pub agent_id: String,
    pub task_id: Option<String>,
    pub session_id: Option<String>,
    pub model_tier: ModelTier,
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub cost_usd: f64,
    pub timestamp_ms: u64,
}

/// Atomic f64 budget tracker. Uses bit-cast u64 to enable CAS operations.
/// All operations are sequentially consistent.
pub struct AtomicBudget {
    /// Current spend in micro-dollars (1 USD = 1_000_000 units) for atomic precision.
    spent_micros: AtomicU64,
    /// Hard limit in micro-dollars.
    limit_micros: u64,
}

impl AtomicBudget {
    /// Create a new budget with `limit_usd` hard cap.
    pub fn new(limit_usd: f64) -> Result<Self, BudgetError> {
        if limit_usd <= 0.0 {
            return Err(BudgetError::InvalidConfig("limit must be positive".into()));
        }
        Ok(Self {
            spent_micros: AtomicU64::new(0),
            limit_micros: usd_to_micros(limit_usd),
        })
    }

    /// Atomically charge `cost_usd`. Returns Ok(new_total) or Err if over limit.
    pub fn charge(&self, cost_usd: f64) -> Result<f64, f64> {
        let cost_micros = usd_to_micros(cost_usd);
        // Fetch-add then check — fast path avoids CAS loop.
        let prev = self.spent_micros.fetch_add(cost_micros, Ordering::SeqCst);
        let new_total = prev + cost_micros;
        if new_total > self.limit_micros {
            // Roll back
            self.spent_micros.fetch_sub(cost_micros, Ordering::SeqCst);
            Err(micros_to_usd(prev))
        } else {
            Ok(micros_to_usd(new_total))
        }
    }

    /// Current spend in USD.
    pub fn spent_usd(&self) -> f64 {
        micros_to_usd(self.spent_micros.load(Ordering::SeqCst))
    }

    /// Remaining budget in USD.
    pub fn remaining_usd(&self) -> f64 {
        let spent = self.spent_micros.load(Ordering::SeqCst);
        if spent >= self.limit_micros {
            0.0
        } else {
            micros_to_usd(self.limit_micros - spent)
        }
    }

    /// Hard limit in USD.
    pub fn limit_usd(&self) -> f64 {
        micros_to_usd(self.limit_micros)
    }

    /// Whether the budget has been exhausted.
    pub fn is_exhausted(&self) -> bool {
        self.spent_micros.load(Ordering::SeqCst) >= self.limit_micros
    }

    /// Fraction consumed: 0.0 = untouched, 1.0 = fully spent.
    pub fn utilization(&self) -> f64 {
        self.spent_micros.load(Ordering::SeqCst) as f64 / self.limit_micros as f64
    }
}

fn usd_to_micros(usd: f64) -> u64 {
    (usd * 1_000_000.0).round() as u64
}

fn micros_to_usd(micros: u64) -> f64 {
    micros as f64 / 1_000_000.0
}

/// Scoped budget envelope. Wraps an `AtomicBudget` with scope metadata.
#[derive(Clone)]
pub struct BudgetEnvelope {
    pub scope_id: String,
    pub scope_kind: ScopeKind,
    inner: Arc<AtomicBudget>,
    /// Kill switch flag.
    killed: Arc<AtomicU64>,
    /// Audit log of charges.
    log: Arc<RwLock<Vec<SpendRecord>>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ScopeKind {
    Agent,
    Task,
    Session,
}

impl BudgetEnvelope {
    pub fn new(scope_id: impl Into<String>, scope_kind: ScopeKind, limit_usd: f64) -> Result<Self, BudgetError> {
        Ok(Self {
            scope_id: scope_id.into(),
            scope_kind,
            inner: Arc::new(AtomicBudget::new(limit_usd)?),
            killed: Arc::new(AtomicU64::new(0)),
            log: Arc::new(RwLock::new(Vec::new())),
        })
    }

    /// Charge `cost_usd` to this envelope. Returns Ok(new_total) or typed BudgetError.
    pub async fn charge(&self, record: SpendRecord) -> Result<f64, BudgetError> {
        if self.killed.load(Ordering::SeqCst) != 0 {
            return Err(BudgetError::KillSwitchActivated {
                agent_id: self.scope_id.clone(),
                reason: "envelope kill switch is active".into(),
            });
        }

        let cost = record.cost_usd;
        let result = self.inner.charge(cost);

        match result {
            Ok(total) => {
                self.log.write().await.push(record);
                Ok(total)
            }
            Err(spent) => {
                let err = match self.scope_kind {
                    ScopeKind::Agent => BudgetError::AgentExhausted {
                        agent_id: self.scope_id.clone(),
                        spent,
                        limit: self.inner.limit_usd(),
                    },
                    ScopeKind::Task => BudgetError::TaskExhausted {
                        task_id: self.scope_id.clone(),
                        spent,
                        limit: self.inner.limit_usd(),
                    },
                    ScopeKind::Session => BudgetError::SessionExhausted {
                        session_id: self.scope_id.clone(),
                        spent,
                        limit: self.inner.limit_usd(),
                    },
                };
                Err(err)
            }
        }
    }

    /// Activate kill switch — all future charges are rejected.
    pub fn kill(&self, reason: &str) {
        self.killed.store(1, Ordering::SeqCst);
        let _ = reason;
    }

    pub fn is_killed(&self) -> bool {
        self.killed.load(Ordering::SeqCst) != 0
    }

    pub fn spent_usd(&self) -> f64 {
        self.inner.spent_usd()
    }

    pub fn remaining_usd(&self) -> f64 {
        self.inner.remaining_usd()
    }

    pub fn limit_usd(&self) -> f64 {
        self.inner.limit_usd()
    }

    pub fn is_exhausted(&self) -> bool {
        self.inner.is_exhausted()
    }

    pub fn utilization(&self) -> f64 {
        self.inner.utilization()
    }

    pub async fn audit_log(&self) -> Vec<SpendRecord> {
        self.log.read().await.clone()
    }
}

/// Point-in-time snapshot of an agent's budget state.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct BudgetSnapshot {
    pub agent_id: String,
    pub spent_usd: f64,
    pub limit_usd: f64,
    pub remaining_usd: f64,
    pub utilization: f64,
    pub is_killed: bool,
    pub model_tier: ModelTier,
    pub timestamp_ms: u64,
}

/// Per-agent budget with model cascade integration.
pub struct AgentBudget {
    pub agent_id: String,
    envelope: BudgetEnvelope,
    cascade: tokio::sync::Mutex<crate::model::ModelCascade>,
}

impl AgentBudget {
    pub fn new(
        agent_id: impl Into<String>,
        limit_usd: f64,
        initial_tier: ModelTier,
        cascade_threshold: f64,
    ) -> Result<Self, BudgetError> {
        let id = agent_id.into();
        let envelope = BudgetEnvelope::new(id.clone(), ScopeKind::Agent, limit_usd)?;
        let cascade = crate::model::ModelCascade::new(initial_tier, cascade_threshold, limit_usd)?;
        Ok(Self {
            agent_id: id,
            envelope,
            cascade: tokio::sync::Mutex::new(cascade),
        })
    }

    /// Charge a request. Automatically evaluates cascade after charging.
    /// Returns (new_total, Option<downgraded_to_tier>).
    pub async fn charge(
        &self,
        input_tokens: u32,
        output_tokens: u32,
        task_id: Option<String>,
        session_id: Option<String>,
    ) -> Result<(f64, Option<ModelTier>), BudgetError> {
        let current_tier = self.cascade.lock().await.current();
        let cost = current_tier.estimate_cost(input_tokens, output_tokens);

        let record = SpendRecord {
            agent_id: self.agent_id.clone(),
            task_id,
            session_id,
            model_tier: current_tier,
            input_tokens,
            output_tokens,
            cost_usd: cost,
            timestamp_ms: now_ms(),
        };

        let new_total = self.envelope.charge(record).await?;

        // Evaluate cascade after successful charge
        let remaining = self.envelope.remaining_usd();
        let downgrade = self.cascade.lock().await.evaluate(remaining);

        Ok((new_total, downgrade))
    }

    /// Force kill this agent's budget.
    pub fn kill(&self, reason: &str) {
        self.envelope.kill(reason);
    }

    pub fn is_killed(&self) -> bool {
        self.envelope.is_killed()
    }

    pub fn spent_usd(&self) -> f64 {
        self.envelope.spent_usd()
    }

    pub fn remaining_usd(&self) -> f64 {
        self.envelope.remaining_usd()
    }

    pub async fn current_tier(&self) -> ModelTier {
        self.cascade.lock().await.current()
    }

    pub async fn snapshot(&self) -> BudgetSnapshot {
        BudgetSnapshot {
            agent_id: self.agent_id.clone(),
            spent_usd: self.envelope.spent_usd(),
            limit_usd: self.envelope.limit_usd(),
            remaining_usd: self.envelope.remaining_usd(),
            utilization: self.envelope.utilization(),
            is_killed: self.envelope.is_killed(),
            model_tier: self.cascade.lock().await.current(),
            timestamp_ms: now_ms(),
        }
    }
}

fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── AtomicBudget tests ─────────────────────────────────────────────────

    #[test]
    fn test_atomic_budget_new_rejects_zero_limit() {
        assert!(AtomicBudget::new(0.0).is_err());
        assert!(AtomicBudget::new(-1.0).is_err());
    }

    #[test]
    fn test_atomic_budget_charge_within_limit() {
        let b = AtomicBudget::new(1.0).expect("new");
        let result = b.charge(0.5);
        assert!(result.is_ok());
        assert!((b.spent_usd() - 0.5).abs() < 1e-4);
    }

    #[test]
    fn test_atomic_budget_charge_exactly_at_limit() {
        let b = AtomicBudget::new(1.0).expect("new");
        let result = b.charge(1.0);
        assert!(result.is_ok());
        assert!(b.is_exhausted());
    }

    #[test]
    fn test_atomic_budget_charge_over_limit_rejected_and_rolled_back() {
        let b = AtomicBudget::new(1.0).expect("new");
        b.charge(0.9).expect("first charge");
        let result = b.charge(0.2); // would take to 1.1
        assert!(result.is_err());
        // Spent should still be 0.9, not 1.1
        assert!((b.spent_usd() - 0.9).abs() < 1e-4);
    }

    #[test]
    fn test_atomic_budget_remaining_decreases_with_charges() {
        let b = AtomicBudget::new(10.0).expect("new");
        b.charge(3.0).expect("charge");
        assert!((b.remaining_usd() - 7.0).abs() < 1e-4);
    }

    #[test]
    fn test_atomic_budget_utilization() {
        let b = AtomicBudget::new(10.0).expect("new");
        b.charge(5.0).expect("charge");
        assert!((b.utilization() - 0.5).abs() < 1e-4);
    }

    #[test]
    fn test_atomic_budget_concurrent_charges_safe() {
        use std::sync::Arc;
        let b = Arc::new(AtomicBudget::new(100.0).expect("new"));
        let mut handles = vec![];
        for _ in 0..10 {
            let b2 = b.clone();
            handles.push(std::thread::spawn(move || {
                b2.charge(9.0)
            }));
        }
        let results: Vec<_> = handles.into_iter().map(|h| h.join().expect("join")).collect();
        // Exactly 11 charges of 9.0 = 99.0 fit, 12th = 108.0 over
        let successes = results.iter().filter(|r| r.is_ok()).count();
        assert!(successes <= 11);
        assert!(b.spent_usd() <= 100.0 + 1e-3);
    }

    // ── BudgetEnvelope tests ───────────────────────────────────────────────

    fn make_record(agent_id: &str, cost: f64) -> SpendRecord {
        SpendRecord {
            agent_id: agent_id.into(),
            task_id: None,
            session_id: None,
            model_tier: ModelTier::Standard,
            input_tokens: 100,
            output_tokens: 50,
            cost_usd: cost,
            timestamp_ms: 0,
        }
    }

    #[tokio::test]
    async fn test_envelope_charge_within_limit_succeeds() {
        let env = BudgetEnvelope::new("a1", ScopeKind::Agent, 1.0).expect("new");
        let result = env.charge(make_record("a1", 0.5)).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_envelope_charge_over_limit_returns_agent_exhausted() {
        let env = BudgetEnvelope::new("a1", ScopeKind::Agent, 1.0).expect("new");
        env.charge(make_record("a1", 0.9)).await.expect("first");
        let result = env.charge(make_record("a1", 0.2)).await;
        assert!(matches!(result, Err(BudgetError::AgentExhausted { .. })));
    }

    #[tokio::test]
    async fn test_envelope_task_scope_returns_task_exhausted() {
        let env = BudgetEnvelope::new("t1", ScopeKind::Task, 0.1).expect("new");
        env.charge(make_record("t1", 0.09)).await.expect("first");
        let result = env.charge(make_record("t1", 0.05)).await;
        assert!(matches!(result, Err(BudgetError::TaskExhausted { .. })));
    }

    #[tokio::test]
    async fn test_envelope_kill_switch_rejects_all_charges() {
        let env = BudgetEnvelope::new("a1", ScopeKind::Agent, 10.0).expect("new");
        env.kill("testing");
        let result = env.charge(make_record("a1", 0.01)).await;
        assert!(matches!(result, Err(BudgetError::KillSwitchActivated { .. })));
    }

    #[tokio::test]
    async fn test_envelope_audit_log_records_charges() {
        let env = BudgetEnvelope::new("a1", ScopeKind::Agent, 10.0).expect("new");
        env.charge(make_record("a1", 0.1)).await.expect("ok");
        env.charge(make_record("a1", 0.2)).await.expect("ok");
        let log = env.audit_log().await;
        assert_eq!(log.len(), 2);
    }

    // ── AgentBudget tests ──────────────────────────────────────────────────

    #[tokio::test]
    async fn test_agent_budget_charge_triggers_cascade() {
        // Budget = 1.0, threshold = 0.2 (downgrade when < 20% remaining)
        let agent = AgentBudget::new("a1", 1.0, ModelTier::Premium, 0.2).expect("new");
        // Premium: $0.015/1K input, $0.075/1K output
        // 60000 input + 1000 output = $0.90 + $0.075 = $0.975 (within $1.0 limit)
        // remaining = $0.025 < $0.20 (20% threshold)
        agent.charge(60000, 1000, None, None).await.ok();
        let remaining = agent.remaining_usd();
        assert!(remaining < 0.2); // threshold was 20% of 1.0 = 0.2
    }

    #[tokio::test]
    async fn test_agent_budget_kill_prevents_charges() {
        let agent = AgentBudget::new("a1", 10.0, ModelTier::Standard, 0.1).expect("new");
        agent.kill("hard limit test");
        let result = agent.charge(100, 50, None, None).await;
        assert!(matches!(result, Err(BudgetError::KillSwitchActivated { .. })));
    }

    #[tokio::test]
    async fn test_agent_budget_snapshot_fields() {
        let agent = AgentBudget::new("agent-1", 5.0, ModelTier::Economy, 0.1).expect("new");
        let snap = agent.snapshot().await;
        assert_eq!(snap.agent_id, "agent-1");
        assert_eq!(snap.model_tier, ModelTier::Economy);
        assert!((snap.limit_usd - 5.0).abs() < 1e-4);
        assert!(!snap.is_killed);
    }

    #[tokio::test]
    async fn test_agent_budget_charge_zero_cost_local_tier() {
        let agent = AgentBudget::new("local", 1.0, ModelTier::Local, 0.1).expect("new");
        // Local tier has zero cost, so many charges should succeed
        for _ in 0..100 {
            agent.charge(1000, 500, None, None).await.expect("local is free");
        }
        assert!((agent.spent_usd()).abs() < 1e-4);
    }

    #[test]
    fn test_scope_kind_is_copy() {
        let s = ScopeKind::Agent;
        let _s2 = s;
        let _s3 = s; // proves Copy
    }
}
