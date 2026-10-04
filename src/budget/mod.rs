//! Core budget primitives: per-agent atomic tracking and scoped envelopes.

use crate::error::BudgetError;
use crate::model::ModelTier;
use crate::pricing::PriceTable;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use tokio::sync::RwLock;

/// One accepted charge, kept in the audit log.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
pub struct SpendRecord {
    /// Agent that was charged.
    pub agent_id: String,
    /// Optional task the charge belongs to.
    pub task_id: Option<String>,
    /// Optional session the charge belongs to.
    pub session_id: Option<String>,
    /// Tier the agent was on when charged.
    pub model_tier: ModelTier,
    /// Exact model name, when the charge was priced from the price table.
    #[serde(default)]
    pub model: Option<String>,
    /// Prompt tokens.
    pub input_tokens: u32,
    /// Completion tokens.
    pub output_tokens: u32,
    /// Cost of this charge in USD.
    pub cost_usd: f64,
    /// Unix time in milliseconds.
    pub timestamp_ms: u64,
}

/// What a successful charge did.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
pub struct ChargeReceipt {
    /// Cost of this charge alone, in USD.
    pub cost_usd: f64,
    /// Total spent by this agent after the charge, in USD.
    pub total_spent_usd: f64,
    /// Set when this charge pushed the agent down to a cheaper tier.
    pub downgraded_to: Option<ModelTier>,
}

/// Lock-free spend counter with a hard limit.
///
/// Amounts are stored as whole nano-dollars (1 USD = 1,000,000,000) in atomics,
/// so concurrent charges from many threads never need a mutex and the limit is
/// never exceeded.
pub struct AtomicBudget {
    spent_nanos: AtomicU64,
    limit_nanos: AtomicU64,
}

impl AtomicBudget {
    /// Create a new budget with a `limit_usd` hard cap.
    ///
    /// # Errors
    /// [`BudgetError::InvalidConfig`] if the limit is not a positive number.
    pub fn new(limit_usd: f64) -> Result<Self, BudgetError> {
        if !limit_usd.is_finite() || limit_usd <= 0.0 {
            return Err(BudgetError::InvalidConfig("limit must be positive".into()));
        }
        Ok(Self {
            spent_nanos: AtomicU64::new(0),
            limit_nanos: AtomicU64::new(usd_to_nanos(limit_usd)),
        })
    }

    /// Atomically charge `cost_usd`. Returns `Ok(new_total)` or `Err(spent_so_far)`
    /// if the charge would cross the limit (nothing is recorded in that case).
    ///
    /// Uses a compare-and-swap loop, so a charge that fits is never refused
    /// because another, larger charge was racing with it.
    pub fn charge(&self, cost_usd: f64) -> Result<f64, f64> {
        let cost = usd_to_nanos(cost_usd);
        loop {
            let spent = self.spent_nanos.load(Ordering::SeqCst);
            let new_total = spent.saturating_add(cost);
            if new_total > self.limit_nanos.load(Ordering::SeqCst) {
                return Err(nanos_to_usd(spent));
            }
            if self
                .spent_nanos
                .compare_exchange_weak(spent, new_total, Ordering::SeqCst, Ordering::SeqCst)
                .is_ok()
            {
                // A concurrent shrink_limit may have lowered the limit between
                // our check and our write. Undo in that case so the cap holds.
                if new_total > self.limit_nanos.load(Ordering::SeqCst) {
                    self.spent_nanos.fetch_sub(cost, Ordering::SeqCst);
                    return Err(nanos_to_usd(spent));
                }
                return Ok(nanos_to_usd(new_total));
            }
        }
    }

    /// Lower the limit by `amount_usd`, but only if the unspent part covers it.
    /// Returns the new limit, or `Err(remaining)` if there is not enough left.
    pub fn shrink_limit(&self, amount_usd: f64) -> Result<f64, f64> {
        let amount = usd_to_nanos(amount_usd);
        loop {
            let limit = self.limit_nanos.load(Ordering::SeqCst);
            let spent = self.spent_nanos.load(Ordering::SeqCst);
            let remaining = limit.saturating_sub(spent);
            if amount > remaining {
                return Err(nanos_to_usd(remaining));
            }
            let new_limit = limit - amount;
            if self
                .limit_nanos
                .compare_exchange_weak(limit, new_limit, Ordering::SeqCst, Ordering::SeqCst)
                .is_ok()
            {
                // Store-then-load on both sides: if a charge slipped in, give the room back.
                if self.spent_nanos.load(Ordering::SeqCst) > new_limit {
                    self.limit_nanos.fetch_add(amount, Ordering::SeqCst);
                    let remaining = self
                        .limit_nanos
                        .load(Ordering::SeqCst)
                        .saturating_sub(self.spent_nanos.load(Ordering::SeqCst));
                    return Err(nanos_to_usd(remaining));
                }
                return Ok(nanos_to_usd(new_limit));
            }
        }
    }

    /// Raise the limit by `amount_usd`. Returns the new limit.
    pub fn grow_limit(&self, amount_usd: f64) -> f64 {
        let amount = usd_to_nanos(amount_usd);
        nanos_to_usd(self.limit_nanos.fetch_add(amount, Ordering::SeqCst).saturating_add(amount))
    }

    /// Current spend in USD.
    pub fn spent_usd(&self) -> f64 {
        nanos_to_usd(self.spent_nanos.load(Ordering::SeqCst))
    }

    /// Remaining budget in USD.
    pub fn remaining_usd(&self) -> f64 {
        let spent = self.spent_nanos.load(Ordering::SeqCst);
        nanos_to_usd(self.limit_nanos.load(Ordering::SeqCst).saturating_sub(spent))
    }

    /// Hard limit in USD.
    pub fn limit_usd(&self) -> f64 {
        nanos_to_usd(self.limit_nanos.load(Ordering::SeqCst))
    }

    /// Whether the budget has been exhausted.
    pub fn is_exhausted(&self) -> bool {
        self.spent_nanos.load(Ordering::SeqCst) >= self.limit_nanos.load(Ordering::SeqCst)
    }

    /// Fraction consumed: 0.0 = untouched, 1.0 = fully spent.
    pub fn utilization(&self) -> f64 {
        let limit = self.limit_nanos.load(Ordering::SeqCst);
        if limit == 0 {
            return 1.0;
        }
        self.spent_nanos.load(Ordering::SeqCst) as f64 / limit as f64
    }
}

pub(crate) fn usd_to_nanos(usd: f64) -> u64 {
    // `as` saturates: negative and NaN become 0, huge values become u64::MAX.
    (usd * 1_000_000_000.0).round() as u64
}

pub(crate) fn nanos_to_usd(nanos: u64) -> f64 {
    nanos as f64 / 1_000_000_000.0
}

/// A budget with a scope (agent, task or session), a kill switch and an audit log.
#[derive(Clone)]
pub struct BudgetEnvelope {
    /// Name of the agent, task or session this envelope belongs to.
    pub scope_id: String,
    /// What kind of scope this is; decides which error a refused charge returns.
    pub scope_kind: ScopeKind,
    inner: Arc<AtomicBudget>,
    killed: Arc<AtomicBool>,
    kill_reason: Arc<Mutex<Option<String>>>,
    log: Arc<RwLock<Vec<SpendRecord>>>,
}

/// The kind of thing a [`BudgetEnvelope`] limits.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
pub enum ScopeKind {
    /// One agent.
    Agent,
    /// One task, possibly shared by several agents.
    Task,
    /// One session.
    Session,
}

impl BudgetEnvelope {
    /// Create an envelope with a hard cap of `limit_usd`.
    ///
    /// # Errors
    /// [`BudgetError::InvalidConfig`] if the limit is not positive.
    pub fn new(scope_id: impl Into<String>, scope_kind: ScopeKind, limit_usd: f64) -> Result<Self, BudgetError> {
        Ok(Self {
            scope_id: scope_id.into(),
            scope_kind,
            inner: Arc::new(AtomicBudget::new(limit_usd)?),
            killed: Arc::new(AtomicBool::new(false)),
            kill_reason: Arc::new(Mutex::new(None)),
            log: Arc::new(RwLock::new(Vec::new())),
        })
    }

    /// Charge `record.cost_usd` to this envelope. Returns the new total, or a typed error.
    ///
    /// # Errors
    /// [`BudgetError::KillSwitchActivated`] after [`kill`](Self::kill), or the
    /// `*Exhausted` variant matching the scope when the cap would be crossed.
    pub async fn charge(&self, record: SpendRecord) -> Result<f64, BudgetError> {
        if self.killed.load(Ordering::SeqCst) {
            return Err(BudgetError::KillSwitchActivated {
                agent_id: self.scope_id.clone(),
                reason: self.kill_reason().unwrap_or_else(|| "kill switch is active".into()),
            });
        }

        match self.inner.charge(record.cost_usd) {
            Ok(total) => {
                self.log.write().await.push(record);
                Ok(total)
            }
            Err(spent) => {
                let limit = self.inner.limit_usd();
                let id = self.scope_id.clone();
                Err(match self.scope_kind {
                    ScopeKind::Agent => BudgetError::AgentExhausted { agent_id: id, spent, limit },
                    ScopeKind::Task => BudgetError::TaskExhausted { task_id: id, spent, limit },
                    ScopeKind::Session => BudgetError::SessionExhausted { session_id: id, spent, limit },
                })
            }
        }
    }

    /// Activate the kill switch: every later charge is refused with `reason`.
    pub fn kill(&self, reason: &str) {
        if let Ok(mut r) = self.kill_reason.lock() {
            *r = Some(reason.to_string());
        }
        self.killed.store(true, Ordering::SeqCst);
    }

    /// The reason given to [`kill`](Self::kill), if the switch is active.
    pub fn kill_reason(&self) -> Option<String> {
        self.kill_reason.lock().ok().and_then(|r| r.clone())
    }

    /// True once [`kill`](Self::kill) has been called.
    pub fn is_killed(&self) -> bool {
        self.killed.load(Ordering::SeqCst)
    }

    /// Spent so far, in USD.
    pub fn spent_usd(&self) -> f64 {
        self.inner.spent_usd()
    }

    /// Left to spend, in USD.
    pub fn remaining_usd(&self) -> f64 {
        self.inner.remaining_usd()
    }

    /// The cap, in USD.
    pub fn limit_usd(&self) -> f64 {
        self.inner.limit_usd()
    }

    /// True when nothing is left to spend.
    pub fn is_exhausted(&self) -> bool {
        self.inner.is_exhausted()
    }

    /// Fraction of the cap spent (0.0 to 1.0).
    pub fn utilization(&self) -> f64 {
        self.inner.utilization()
    }

    /// Every accepted charge, oldest first.
    pub async fn audit_log(&self) -> Vec<SpendRecord> {
        self.log.read().await.clone()
    }

    pub(crate) fn atomic(&self) -> &AtomicBudget {
        &self.inner
    }
}

/// Point-in-time snapshot of an agent's budget state.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
pub struct BudgetSnapshot {
    /// Agent name.
    pub agent_id: String,
    /// Spent so far, in USD.
    pub spent_usd: f64,
    /// Cap, in USD.
    pub limit_usd: f64,
    /// Left to spend, in USD.
    pub remaining_usd: f64,
    /// Fraction of the cap spent.
    pub utilization: f64,
    /// Whether the kill switch is on.
    pub is_killed: bool,
    /// Tier the agent is on now.
    pub model_tier: ModelTier,
    /// Unix time in milliseconds when the snapshot was taken.
    pub timestamp_ms: u64,
}

/// Per-agent budget with an automatic downgrade cascade.
pub struct AgentBudget {
    /// Agent name.
    pub agent_id: String,
    envelope: BudgetEnvelope,
    cascade: tokio::sync::Mutex<crate::model::ModelCascade>,
}

impl AgentBudget {
    /// Create an agent with a `limit_usd` cap, starting on `initial_tier`, that
    /// steps down one tier whenever the remaining fraction of its budget is at
    /// or below `cascade_threshold` (0.0 to 1.0).
    ///
    /// # Errors
    /// [`BudgetError::InvalidConfig`] for a non-positive limit or a threshold outside 0..=1.
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

    /// Charge a request priced by the agent's current [`ModelTier`] (a rough
    /// average). Returns `(total_spent, Some(tier))` when the agent was just
    /// downgraded.
    ///
    /// Prefer [`charge_model`](Self::charge_model) when you know the model name.
    ///
    /// # Errors
    /// See [`BudgetEnvelope::charge`].
    pub async fn charge(
        &self,
        input_tokens: u32,
        output_tokens: u32,
        task_id: Option<String>,
        session_id: Option<String>,
    ) -> Result<(f64, Option<ModelTier>), BudgetError> {
        let r = self.charge_inner(None, None, input_tokens, output_tokens, task_id, session_id).await?;
        Ok((r.total_spent_usd, r.downgraded_to))
    }

    /// Charge a call to a named model at that model's real price from the
    /// built-in [`PriceTable`].
    ///
    /// # Errors
    /// [`BudgetError::UnknownModel`] if the model is not in the table, otherwise
    /// see [`BudgetEnvelope::charge`].
    pub async fn charge_model(
        &self,
        model: &str,
        input_tokens: u32,
        output_tokens: u32,
        task_id: Option<String>,
        session_id: Option<String>,
    ) -> Result<ChargeReceipt, BudgetError> {
        self.charge_model_with(PriceTable::builtin(), model, input_tokens, output_tokens, task_id, session_id)
            .await
    }

    /// Like [`charge_model`](Self::charge_model) with your own [`PriceTable`].
    ///
    /// # Errors
    /// Same as [`charge_model`](Self::charge_model).
    pub async fn charge_model_with(
        &self,
        table: &PriceTable,
        model: &str,
        input_tokens: u32,
        output_tokens: u32,
        task_id: Option<String>,
        session_id: Option<String>,
    ) -> Result<ChargeReceipt, BudgetError> {
        let cost = table.require(model)?.cost(u64::from(input_tokens), u64::from(output_tokens));
        self.charge_inner(Some(model), Some(cost), input_tokens, output_tokens, task_id, session_id)
            .await
    }

    /// Charge a call from the usage block the provider returned (see
    /// [`crate::usage::Usage`]), priced with prompt-cache discounts from the
    /// built-in [`PriceTable`]. This is the most accurate way to charge.
    ///
    /// # Errors
    /// [`BudgetError::UnknownModel`], otherwise see [`BudgetEnvelope::charge`].
    pub async fn charge_usage(
        &self,
        model: &str,
        usage: &crate::usage::Usage,
        task_id: Option<String>,
        session_id: Option<String>,
    ) -> Result<ChargeReceipt, BudgetError> {
        let cost = PriceTable::builtin().require(model)?.cost_usage(usage);
        let clamp = |n: u64| u32::try_from(n).unwrap_or(u32::MAX);
        self.charge_inner(Some(model), Some(cost), clamp(usage.input_tokens), clamp(usage.output_tokens), task_id, session_id)
            .await
    }

    /// Charge an exact dollar amount (for example the cost your provider reported).
    ///
    /// # Errors
    /// [`BudgetError::InvalidConfig`] for a negative or non-finite amount,
    /// otherwise see [`BudgetEnvelope::charge`].
    pub async fn charge_usd(
        &self,
        cost_usd: f64,
        task_id: Option<String>,
        session_id: Option<String>,
    ) -> Result<ChargeReceipt, BudgetError> {
        if !cost_usd.is_finite() || cost_usd < 0.0 {
            return Err(BudgetError::InvalidConfig(format!("charge must be >= 0, got {cost_usd}")));
        }
        self.charge_inner(None, Some(cost_usd), 0, 0, task_id, session_id).await
    }

    pub(crate) async fn charge_inner(
        &self,
        model: Option<&str>,
        cost: Option<f64>,
        input_tokens: u32,
        output_tokens: u32,
        task_id: Option<String>,
        session_id: Option<String>,
    ) -> Result<ChargeReceipt, BudgetError> {
        let current_tier = self.cascade.lock().await.current();
        let cost = cost.unwrap_or_else(|| current_tier.estimate_cost(input_tokens, output_tokens));
        // Round the same way the atomic counter does, so receipts add up exactly.
        let cost = nanos_to_usd(usd_to_nanos(cost));

        let record = SpendRecord {
            agent_id: self.agent_id.clone(),
            task_id,
            session_id,
            model_tier: current_tier,
            model: model.map(str::to_string),
            input_tokens,
            output_tokens,
            cost_usd: cost,
            timestamp_ms: now_ms(),
        };

        let total = match self.envelope.charge(record).await {
            Ok(total) => total,
            Err(e) => {
                let killed = matches!(e, BudgetError::KillSwitchActivated { .. });
                crate::observe::refused(&self.agent_id, killed, &e.to_string());
                return Err(e);
            }
        };
        let remaining = self.envelope.remaining_usd();
        crate::observe::charged(&self.agent_id, model, cost, total, remaining);
        let downgraded_to = self.cascade.lock().await.evaluate(remaining);
        if let Some(tier) = downgraded_to {
            crate::observe::downgraded(&self.agent_id, tier);
        }
        Ok(ChargeReceipt { cost_usd: cost, total_spent_usd: total, downgraded_to })
    }

    /// Force kill this agent's budget.
    pub fn kill(&self, reason: &str) {
        crate::observe::killed(&self.agent_id, reason);
        self.envelope.kill(reason);
    }

    /// True once the agent has been killed.
    pub fn is_killed(&self) -> bool {
        self.envelope.is_killed()
    }

    /// Spent so far, in USD.
    pub fn spent_usd(&self) -> f64 {
        self.envelope.spent_usd()
    }

    /// Left to spend, in USD.
    pub fn remaining_usd(&self) -> f64 {
        self.envelope.remaining_usd()
    }

    /// The agent's cap, in USD.
    pub fn limit_usd(&self) -> f64 {
        self.envelope.limit_usd()
    }

    /// The tier the agent should use for its next call.
    pub async fn current_tier(&self) -> ModelTier {
        self.cascade.lock().await.current()
    }

    /// Every accepted charge, oldest first.
    pub async fn audit_log(&self) -> Vec<SpendRecord> {
        self.envelope.audit_log().await
    }

    /// Move `amount_usd` of unspent cap out of this agent. Returns the new cap.
    pub(crate) async fn shrink_limit(&self, amount_usd: f64) -> Result<f64, f64> {
        let new_limit = self.envelope.atomic().shrink_limit(amount_usd)?;
        self.cascade.lock().await.set_total_budget(new_limit);
        Ok(new_limit)
    }

    /// Add `amount_usd` to this agent's cap. Returns the new cap.
    pub(crate) async fn grow_limit(&self, amount_usd: f64) -> f64 {
        let new_limit = self.envelope.atomic().grow_limit(amount_usd);
        self.cascade.lock().await.set_total_budget(new_limit);
        new_limit
    }

    /// A point-in-time copy of this agent's numbers.
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

pub(crate) fn now_ms() -> u64 {
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
            model: None,
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

    // ── Regression tests for 0.2.0 fixes ─────────────────────────────────────

    #[test]
    fn test_cas_charge_never_refuses_a_fitting_charge() {
        // Before 0.2.0 a charge was added first and rolled back if it crossed the
        // limit, so a large failing charge could make a small fitting charge on
        // another thread fail too. With CAS every 1-cent charge must succeed.
        use std::sync::Arc;
        for _ in 0..50 {
            let b = Arc::new(AtomicBudget::new(1.0).expect("new"));
            b.charge(0.5).expect("prefill");
            let mut handles = vec![];
            for i in 0..8 {
                let b2 = b.clone();
                handles.push(std::thread::spawn(move || {
                    let mut small_ok = 0;
                    for _ in 0..500 {
                        if i % 2 == 0 {
                            let _ = b2.charge(0.6); // never fits
                        } else if b2.charge(0.0).is_ok() {
                            small_ok += 1;
                        }
                    }
                    small_ok
                }));
            }
            let ok: usize = handles.into_iter().map(|h| h.join().expect("join")).sum();
            assert_eq!(ok, 4 * 500, "a zero-cost charge was refused");
            assert!((b.spent_usd() - 0.5).abs() < 1e-9);
        }
    }

    #[test]
    fn test_shrink_and_grow_limit() {
        let b = AtomicBudget::new(10.0).expect("new");
        b.charge(4.0).expect("charge");
        assert!(b.shrink_limit(7.0).is_err()); // only 6 left
        assert_eq!(b.shrink_limit(6.0), Ok(4.0));
        assert!(b.charge(0.01).is_err());
        assert!((b.grow_limit(1.0) - 5.0).abs() < 1e-9);
        assert!(b.charge(1.0).is_ok());
    }

    #[tokio::test]
    async fn test_kill_reason_is_reported() {
        let agent = AgentBudget::new("a1", 10.0, ModelTier::Standard, 0.1).expect("new");
        agent.kill("runaway loop");
        match agent.charge(1, 1, None, None).await {
            Err(BudgetError::KillSwitchActivated { reason, .. }) => assert_eq!(reason, "runaway loop"),
            other => panic!("expected kill switch error, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_charge_model_uses_real_price() {
        let agent = AgentBudget::new("a1", 10.0, ModelTier::Premium, 0.1).expect("new");
        let r = agent.charge_model("gpt-4o-mini", 1_000_000, 0, None, None).await.expect("charge");
        // gpt-4o-mini is $0.15 per million input tokens; the Premium tier guess would be $15.
        assert!((r.cost_usd - 0.15).abs() < 1e-9, "{}", r.cost_usd);
        assert!((agent.spent_usd() - 0.15).abs() < 1e-9);
        let log = agent.audit_log().await;
        assert_eq!(log[0].model.as_deref(), Some("gpt-4o-mini"));
        assert!(matches!(
            agent.charge_model("not-a-model", 1, 1, None, None).await,
            Err(BudgetError::UnknownModel(_))
        ));
    }

    #[tokio::test]
    async fn test_charge_usd_rejects_negative() {
        let agent = AgentBudget::new("a1", 1.0, ModelTier::Standard, 0.1).expect("new");
        assert!(agent.charge_usd(-1.0, None, None).await.is_err());
        assert!(agent.charge_usd(f64::NAN, None, None).await.is_err());
        let r = agent.charge_usd(0.25, None, None).await.expect("ok");
        assert!((r.total_spent_usd - 0.25).abs() < 1e-9);
    }

    #[tokio::test]
    async fn test_tiny_charges_are_not_rounded_away() {
        // 0.1.x counted whole micro-dollars, so a 1-token gpt-4o-mini call
        // ($0.00000015) was rounded to $0 and never counted.
        let agent = AgentBudget::new("a1", 1.0, ModelTier::Standard, 0.1).expect("new");
        for _ in 0..1_000 {
            agent.charge_model("gpt-4o-mini", 1, 0, None, None).await.expect("charge");
        }
        assert!((agent.spent_usd() - 0.00015).abs() < 1e-12, "{}", agent.spent_usd());
    }

    #[test]
    fn test_public_types_are_send_and_sync() {
        fn check<T: Send + Sync>() {}
        check::<AtomicBudget>();
        check::<BudgetEnvelope>();
        check::<AgentBudget>();
        check::<crate::FleetAllocator>();
        check::<crate::BudgetStore>();
        check::<crate::PriceTable>();
        check::<BudgetError>();
    }

    #[cfg(feature = "schemars")]
    #[test]
    fn test_json_schemas_describe_the_records() {
        let schema = serde_json::to_value(schemars::schema_for!(SpendRecord)).unwrap();
        let props = schema["properties"].as_object().unwrap();
        for field in ["agent_id", "model", "cost_usd", "timestamp_ms"] {
            assert!(props.contains_key(field), "missing {field}");
        }
        let snap = serde_json::to_value(schemars::schema_for!(crate::persistence::PersistedBudgetState)).unwrap();
        assert!(snap["properties"]["snapshots"].is_object());
    }

    #[tokio::test]
    async fn test_charge_usage_applies_cache_discount() {
        let agent = AgentBudget::new("a", 10.0, ModelTier::Standard, 0.1).expect("new");
        // OpenAI: 1,000,000 prompt tokens of which 900,000 cached, gpt-4o
        // ($2.50 in, $1.25 cached per million).
        let u = crate::usage::Usage::from_json(
            r#"{"prompt_tokens":1000000,"completion_tokens":0,"prompt_tokens_details":{"cached_tokens":900000}}"#,
        )
        .expect("usage");
        let r = agent.charge_usage("gpt-4o", &u, None, None).await.expect("charge");
        assert!((r.cost_usd - (0.25 + 1.125)).abs() < 1e-9, "{}", r.cost_usd);
    }
}
