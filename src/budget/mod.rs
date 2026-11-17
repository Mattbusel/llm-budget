//! Core budget primitives: per-agent atomic tracking and scoped envelopes.
//!
//! # Authorize, then settle
//!
//! An LLM call is paid for the moment the provider answers, so a budget has
//! two separate jobs:
//!
//! 1. **Before the call**, decide whether it may happen at all. Use
//!    [`AgentBudget::reserve`] with an estimate: it holds that much headroom
//!    with a compare-and-swap, so N agents racing for the last dollar cannot
//!    all be admitted. A refused reservation means the call is never made.
//! 2. **After the call**, book what it really cost with
//!    [`Reservation::settle`] (or [`AgentBudget::charge_usd_force`] when nothing
//!    was reserved). Settlement always records the spend, even past the cap,
//!    because the money is already gone; [`ChargeOutcome::over_cap`] tells you
//!    to stop.
//!
//! This keeps [`AgentBudget::spent_usd`] equal to what the provider will bill,
//! and the sum of the [audit log](AgentBudget::audit_log) equal to
//! `spent_usd` (as long as the log keeps every record, see [`AuditLog`]).
//!
//! The older `charge*` methods remain: they check and book in one step and
//! refuse (booking nothing) when the charge would cross the cap. They suit
//! calls you price *before* making them.
//!
//! ```
//! use llm_budget::{AgentBudget, ChargeContext, ModelTier};
//!
//! # fn main() -> Result<(), llm_budget::BudgetError> {
//! let agent = AgentBudget::new("researcher", 1.00, ModelTier::Standard, 0.2)?;
//!
//! // Hold $0.30 while the call is in flight.
//! let hold = agent.reserve(0.30)?;
//! assert!((agent.remaining_usd() - 0.70).abs() < 1e-9);
//!
//! // The provider says the call cost $0.25: book exactly that.
//! let outcome = hold.settle_with(0.25, ChargeContext::new().model("gpt-4o").tokens(80_000, 5_000));
//! assert!(!outcome.over_cap);
//! assert!((agent.spent_usd() - 0.25).abs() < 1e-9);
//!
//! // A call that ran long is still booked in full, and flagged.
//! let outcome = agent.charge_usd_force(0.90, ChargeContext::new().task("t-7"));
//! assert!(outcome.over_cap);
//! assert!((agent.spent_usd() - 1.15).abs() < 1e-9);
//! assert!(agent.reserve(0.01).is_err()); // nothing left to hold
//! # Ok(())
//! # }
//! ```

use crate::error::BudgetError;
use crate::model::{ModelCascade, ModelTier};
use crate::pricing::{PriceAction, PriceStatus, PriceTable, UnpricedPolicy};
use std::collections::VecDeque;

pub mod scope;
pub use scope::{Limit, ScopeGuard, ScopeReceipt, DEFAULT_MAX_DEPTH};

/// Any [`AgentBudget`] is a budget scope: call [`AgentBudget::child`] on it
/// to carve out a child scope. The alias names the role.
pub type BudgetScope = AgentBudget;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

/// One booked charge, kept in the audit log.
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

/// What a settlement ([`Reservation::settle`], [`AgentBudget::charge_usd_force`])
/// booked. Settlement never fails: the spend is always recorded.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[non_exhaustive]
pub struct ChargeOutcome {
    /// What was booked for this charge, in USD (rounded to whole nano-dollars).
    pub recorded_usd: f64,
    /// Total spent by this agent after the charge, in USD. May exceed the cap.
    pub total_spent_usd: f64,
    /// True when, after this charge, spend plus in-flight reservations is
    /// above the cap. Stop making calls on this budget.
    pub over_cap: bool,
    /// Set when this charge pushed the agent down to a cheaper tier.
    pub downgraded_to: Option<ModelTier>,
}

/// Who and what a charge was for, recorded in its [`SpendRecord`].
///
/// ```
/// use llm_budget::ChargeContext;
/// let ctx = ChargeContext::new().task("t-1").session("s-9").model("gpt-4o").tokens(1_200, 300);
/// assert_eq!(ctx.task_id.as_deref(), Some("t-1"));
/// ```
#[derive(Debug, Clone, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct ChargeContext {
    /// Task the charge belongs to.
    pub task_id: Option<String>,
    /// Session the charge belongs to.
    pub session_id: Option<String>,
    /// Model that was called.
    pub model: Option<String>,
    /// Prompt tokens.
    pub input_tokens: u32,
    /// Completion tokens.
    pub output_tokens: u32,
}

impl ChargeContext {
    /// An empty context.
    pub fn new() -> Self {
        Self::default()
    }

    /// Attach a task id.
    pub fn task(mut self, task_id: impl Into<String>) -> Self {
        self.task_id = Some(task_id.into());
        self
    }

    /// Attach a session id.
    pub fn session(mut self, session_id: impl Into<String>) -> Self {
        self.session_id = Some(session_id.into());
        self
    }

    /// Record which model was called.
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    /// Record the token counts (saturating at `u32::MAX`).
    pub fn tokens(mut self, input_tokens: u64, output_tokens: u64) -> Self {
        self.input_tokens = u32::try_from(input_tokens).unwrap_or(u32::MAX);
        self.output_tokens = u32::try_from(output_tokens).unwrap_or(u32::MAX);
        self
    }

    /// Record the token counts of a provider-reported [`Usage`](crate::Usage)
    /// (all prompt tokens as input).
    pub fn usage(self, usage: &crate::usage::Usage) -> Self {
        self.tokens(usage.input_tokens, usage.output_tokens)
    }
}

/// How much of the audit log an envelope keeps in memory.
///
/// Every booked charge produces one [`SpendRecord`]. A long-running service
/// would grow an unbounded log forever, so the default keeps the most recent
/// 1,024 records. Use [`drain_audit_log`](AgentBudget::drain_audit_log) to
/// ship records elsewhere (a database, a file) before they are evicted, and
/// [`audit_records_dropped`](AgentBudget::audit_records_dropped) to see whether
/// any were.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
pub enum AuditLog {
    /// Keep no records.
    None,
    /// Keep the most recent `n` records; older ones are evicted.
    Ring(usize),
    /// Keep every record (memory grows with every charge).
    Unbounded,
}

impl Default for AuditLog {
    /// `Ring(1024)`.
    fn default() -> Self {
        AuditLog::Ring(1024)
    }
}

#[derive(Debug, Default)]
struct AuditBuffer {
    mode: AuditLog,
    records: VecDeque<SpendRecord>,
    dropped: u64,
}

impl AuditBuffer {
    fn new(mode: AuditLog) -> Self {
        let mut b = Self { mode, ..Self::default() };
        b.trim();
        b
    }

    fn push(&mut self, record: SpendRecord) {
        self.records.push_back(record);
        self.trim();
    }

    fn trim(&mut self) {
        let cap = match self.mode {
            AuditLog::None => 0,
            AuditLog::Ring(n) => n,
            AuditLog::Unbounded => return,
        };
        while self.records.len() > cap {
            self.records.pop_front();
            self.dropped += 1;
        }
    }
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    // Every critical section here is a few field updates that cannot leave the
    // data half-written, so a poisoned lock is still safe to use.
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

/// Lock-free spend counter with a hard limit.
///
/// Amounts are stored as whole nano-dollars (1 USD = 1,000,000,000) in atomics,
/// so concurrent charges from many threads never need a mutex. [`charge`](Self::charge)
/// never takes the counter past the limit; [`charge_force`](Self::charge_force)
/// books spend that already happened and may.
///
/// Headroom held by in-flight reservations ([`AgentBudget::reserve`]) counts
/// against the limit, so it is not available to other charges, but it is not
/// part of [`spent_usd`](Self::spent_usd) until it is settled.
pub struct AtomicBudget {
    /// Settled spend plus in-flight reservations. The limit is enforced on this.
    committed_nanos: AtomicU64,
    /// In-flight reservations (a part of `committed_nanos`).
    reserved_nanos: AtomicU64,
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
        Ok(Self::from_nanos(usd_to_nanos(limit_usd)))
    }

    /// A budget with a limit of exactly `limit_nanos` nano-dollars.
    pub(crate) fn from_nanos(limit_nanos: u64) -> Self {
        Self {
            committed_nanos: AtomicU64::new(0),
            reserved_nanos: AtomicU64::new(0),
            limit_nanos: AtomicU64::new(limit_nanos),
        }
    }

    /// In-flight reservations, in nano-dollars.
    pub(crate) fn reserved_nanos(&self) -> u64 {
        self.reserved_nanos.load(Ordering::SeqCst)
    }

    /// Settled spend plus in-flight reservations, in nano-dollars.
    pub(crate) fn committed_nanos(&self) -> u64 {
        self.committed_nanos.load(Ordering::SeqCst)
    }

    /// The limit, in nano-dollars.
    pub(crate) fn limit_nanos(&self) -> u64 {
        self.limit_nanos.load(Ordering::SeqCst)
    }

    /// Atomically charge `cost_usd`. Returns `Ok(new_total)` or `Err(spent_so_far)`
    /// if the charge would cross the limit (nothing is recorded in that case).
    ///
    /// Uses a compare-and-swap loop, so a charge that fits is never refused
    /// because another, larger charge was racing with it.
    pub fn charge(&self, cost_usd: f64) -> Result<f64, f64> {
        match self.try_commit(usd_to_nanos(cost_usd)) {
            Ok(committed) => Ok(self.settled_usd(committed)),
            Err(committed) => Err(self.settled_usd(committed)),
        }
    }

    /// Book `cost_usd` unconditionally, even past the limit (saturating at
    /// `u64::MAX` nano-dollars). Use it for spend that has already happened.
    /// Negative, NaN and infinite amounts book nothing. Returns the new total.
    pub fn charge_force(&self, cost_usd: f64) -> f64 {
        let committed = self.settle_nanos(0, usd_to_nanos(sanitize_usd(cost_usd)));
        self.settled_usd(committed)
    }

    /// Settled spend for a committed total read a moment ago.
    fn settled_usd(&self, committed: u64) -> f64 {
        nanos_to_usd(committed.saturating_sub(self.reserved_nanos.load(Ordering::SeqCst)))
    }

    /// CAS-add `nanos` to the committed counter if it stays within the limit.
    /// Returns the new committed total or the committed total that blocked it.
    fn try_commit(&self, nanos: u64) -> Result<u64, u64> {
        loop {
            let committed = self.committed_nanos.load(Ordering::SeqCst);
            let new_total = committed.saturating_add(nanos);
            if new_total > self.limit_nanos.load(Ordering::SeqCst) {
                return Err(committed);
            }
            if self
                .committed_nanos
                .compare_exchange_weak(committed, new_total, Ordering::SeqCst, Ordering::SeqCst)
                .is_ok()
            {
                // A concurrent shrink_limit may have lowered the limit between
                // our check and our write. Undo in that case so the cap holds.
                if new_total > self.limit_nanos.load(Ordering::SeqCst) {
                    self.committed_nanos.fetch_sub(nanos, Ordering::SeqCst);
                    return Err(committed);
                }
                return Ok(new_total);
            }
        }
    }

    /// Hold `nanos` of headroom. Returns `Err(spent_usd)` if it does not fit.
    pub(crate) fn reserve_nanos(&self, nanos: u64) -> Result<(), f64> {
        // Reserved is raised after committed, and lowered before it, so
        // `committed >= reserved` always holds.
        self.try_commit(nanos).map_err(|_| self.spent_usd())?;
        self.reserved_nanos.fetch_add(nanos, Ordering::SeqCst);
        Ok(())
    }

    /// Release a reservation of `reserved` nanos and book `actual` nanos in its
    /// place, in one update of the committed counter. Returns the committed
    /// total afterwards.
    pub(crate) fn settle_nanos(&self, reserved: u64, actual: u64) -> u64 {
        if reserved > 0 {
            sub_saturating(&self.reserved_nanos, reserved);
        }
        let mut current = self.committed_nanos.load(Ordering::SeqCst);
        loop {
            let next = current.saturating_sub(reserved).saturating_add(actual);
            match self
                .committed_nanos
                .compare_exchange_weak(current, next, Ordering::SeqCst, Ordering::SeqCst)
            {
                Ok(_) => return next,
                Err(seen) => current = seen,
            }
        }
    }

    /// True when the committed total is above the limit.
    pub(crate) fn over_limit(&self, committed: u64) -> bool {
        committed > self.limit_nanos.load(Ordering::SeqCst)
    }

    /// Lower the limit by `amount_usd`, but only if the unspent, unreserved
    /// part covers it. Returns the new limit, or `Err(remaining)` if there is
    /// not enough left.
    pub fn shrink_limit(&self, amount_usd: f64) -> Result<f64, f64> {
        let amount = usd_to_nanos(amount_usd);
        loop {
            let limit = self.limit_nanos.load(Ordering::SeqCst);
            let committed = self.committed_nanos.load(Ordering::SeqCst);
            let remaining = limit.saturating_sub(committed);
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
                if self.committed_nanos.load(Ordering::SeqCst) > new_limit {
                    self.limit_nanos.fetch_add(amount, Ordering::SeqCst);
                    let remaining = self
                        .limit_nanos
                        .load(Ordering::SeqCst)
                        .saturating_sub(self.committed_nanos.load(Ordering::SeqCst));
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

    /// Settled spend in USD (in-flight reservations are not included).
    ///
    /// Exact whenever no reservation is being taken or settled at the same
    /// moment; while one is, a reading can be off by that reservation.
    pub fn spent_usd(&self) -> f64 {
        let reserved = self.reserved_nanos.load(Ordering::SeqCst);
        let committed = self.committed_nanos.load(Ordering::SeqCst);
        nanos_to_usd(committed.saturating_sub(reserved))
    }

    /// Headroom held by in-flight reservations, in USD.
    pub fn reserved_usd(&self) -> f64 {
        nanos_to_usd(self.reserved_nanos.load(Ordering::SeqCst))
    }

    /// Remaining budget in USD: the limit minus spend and in-flight reservations.
    pub fn remaining_usd(&self) -> f64 {
        let committed = self.committed_nanos.load(Ordering::SeqCst);
        nanos_to_usd(self.limit_nanos.load(Ordering::SeqCst).saturating_sub(committed))
    }

    /// Hard limit in USD.
    pub fn limit_usd(&self) -> f64 {
        nanos_to_usd(self.limit_nanos.load(Ordering::SeqCst))
    }

    /// Whether nothing is left to spend or reserve.
    pub fn is_exhausted(&self) -> bool {
        self.committed_nanos.load(Ordering::SeqCst) >= self.limit_nanos.load(Ordering::SeqCst)
    }

    /// Fraction consumed (spend plus reservations): 0.0 = untouched, 1.0 =
    /// fully spent. Can exceed 1.0 after a forced charge.
    pub fn utilization(&self) -> f64 {
        let limit = self.limit_nanos.load(Ordering::SeqCst);
        if limit == 0 {
            return 1.0;
        }
        self.committed_nanos.load(Ordering::SeqCst) as f64 / limit as f64
    }
}

fn sub_saturating(a: &AtomicU64, n: u64) {
    let mut current = a.load(Ordering::SeqCst);
    loop {
        match a.compare_exchange_weak(current, current.saturating_sub(n), Ordering::SeqCst, Ordering::SeqCst) {
            Ok(_) => return,
            Err(seen) => current = seen,
        }
    }
}

pub(crate) fn usd_to_nanos(usd: f64) -> u64 {
    // `as` saturates: negative and NaN become 0, huge values become u64::MAX.
    (usd * 1_000_000_000.0).round() as u64
}

pub(crate) fn nanos_to_usd(nanos: u64) -> f64 {
    nanos as f64 / 1_000_000_000.0
}

/// Negative, NaN and infinite amounts become 0.
fn sanitize_usd(usd: f64) -> f64 {
    if usd.is_finite() && usd > 0.0 {
        usd
    } else {
        0.0
    }
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
    log: Arc<Mutex<AuditBuffer>>,
    /// Set for a child scope: the hold it draws on in its parent.
    parent: Option<Arc<scope::ParentLink>>,
    /// Set once a child scope is settled or dropped.
    closed: Arc<AtomicBool>,
    /// Spend booked here that was forwarded from descendant scopes, in nanos.
    child_nanos: Arc<AtomicU64>,
    /// Child scopes created directly under this one.
    delegations: Arc<AtomicU64>,
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
    /// Create an envelope with a hard cap of `limit_usd` and the default
    /// [`AuditLog`] (the last 1,024 records).
    ///
    /// # Errors
    /// [`BudgetError::InvalidConfig`] if the limit is not positive.
    pub fn new(scope_id: impl Into<String>, scope_kind: ScopeKind, limit_usd: f64) -> Result<Self, BudgetError> {
        Ok(Self::with_atomic(scope_id.into(), scope_kind, AtomicBudget::new(limit_usd)?, AuditLog::default(), None))
    }

    fn with_atomic(
        scope_id: String,
        scope_kind: ScopeKind,
        inner: AtomicBudget,
        audit: AuditLog,
        parent: Option<Arc<scope::ParentLink>>,
    ) -> Self {
        Self {
            scope_id,
            scope_kind,
            inner: Arc::new(inner),
            killed: Arc::new(AtomicBool::new(false)),
            kill_reason: Arc::new(Mutex::new(None)),
            log: Arc::new(Mutex::new(AuditBuffer::new(audit))),
            parent,
            closed: Arc::new(AtomicBool::new(false)),
            child_nanos: Arc::new(AtomicU64::new(0)),
            delegations: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Keep the audit log as `mode` says (trims what is already there).
    ///
    /// Clones of an envelope share one audit log, so this changes the mode
    /// (and trims the records) for every clone, not just the value it is
    /// called on. Set it once, right after [`new`](Self::new) and before the
    /// envelope is cloned or shared.
    pub fn with_audit_log(self, mode: AuditLog) -> Self {
        {
            let mut log = lock(&self.log);
            log.mode = mode;
            log.trim();
        }
        self
    }

    /// Charge `record.cost_usd` to this envelope. Returns the new total, or a typed error.
    ///
    /// # Errors
    /// [`BudgetError::KillSwitchActivated`] after [`kill`](Self::kill), or the
    /// `*Exhausted` variant matching the scope when the cap would be crossed.
    pub async fn charge(&self, record: SpendRecord) -> Result<f64, BudgetError> {
        self.check_alive()?;
        let nanos = usd_to_nanos(record.cost_usd);
        match self.inner.charge(record.cost_usd) {
            Ok(total) => {
                self.forward(&record, nanos);
                lock(&self.log).push(record);
                Ok(total)
            }
            Err(spent) => Err(self.exhausted(spent)),
        }
    }

    /// Book `record.cost_usd` whatever the cap or the kill switch say, and log
    /// it: for spend that has already happened. Negative, NaN and infinite
    /// amounts book $0.
    pub fn charge_force(&self, mut record: SpendRecord) -> ChargeOutcome {
        record.cost_usd = nanos_to_usd(usd_to_nanos(sanitize_usd(record.cost_usd)));
        let recorded_usd = record.cost_usd;
        let (total_spent_usd, over_cap) = self.book(record, 0);
        ChargeOutcome { recorded_usd, total_spent_usd, over_cap, downgraded_to: None }
    }

    /// Release `reserved` nanos and book the record's cost in their place.
    /// Returns the settled total and whether the cap is now crossed.
    fn book(&self, record: SpendRecord, reserved: u64) -> (f64, bool) {
        let actual = usd_to_nanos(sanitize_usd(record.cost_usd));
        // Forward first: while the reservation is still open here, a
        // concurrent trim of a closed scope keeps enough hold to cover it.
        self.forward(&record, actual);
        let committed = self.inner.settle_nanos(reserved, actual);
        lock(&self.log).push(record);
        self.trim_if_closed();
        (self.inner.settled_usd(committed), self.inner.over_limit(committed))
    }

    /// Book spend forwarded from a child scope: `actual` nanos replace
    /// `from_hold` nanos of the child's hold here.
    pub(crate) fn book_from_child(&self, record: &SpendRecord, from_hold: u64, actual: u64) {
        self.forward(record, actual);
        self.inner.settle_nanos(from_hold, actual);
        self.child_nanos.fetch_add(actual, Ordering::SeqCst);
        lock(&self.log).push(record.clone());
        self.trim_if_closed();
    }

    /// A child scope's spend is booked on every ancestor as it happens.
    fn forward(&self, record: &SpendRecord, actual: u64) {
        if let Some(link) = &self.parent {
            link.forward(record, actual);
        }
    }

    /// Release `nanos` of reservation without booking anything.
    pub(crate) fn release(&self, nanos: u64) {
        if nanos > 0 {
            self.inner.settle_nanos(nanos, 0);
            self.trim_if_closed();
        }
    }

    /// A closed scope gives its parent back every part of its hold that no
    /// in-flight reservation still needs.
    fn trim_if_closed(&self) {
        if let Some(link) = &self.parent {
            if self.closed.load(Ordering::SeqCst) {
                link.trim(&self.inner);
            }
        }
    }

    fn check_alive(&self) -> Result<(), BudgetError> {
        if self.closed.load(Ordering::SeqCst) {
            return Err(BudgetError::ScopeClosed { scope_id: self.scope_id.clone() });
        }
        if self.killed.load(Ordering::SeqCst) {
            return Err(BudgetError::KillSwitchActivated {
                agent_id: self.scope_id.clone(),
                reason: self.kill_reason().unwrap_or_else(|| "kill switch is active".into()),
            });
        }
        // A child scope stops when any ancestor is killed or closed.
        if let Some(link) = &self.parent {
            link.parent().check_alive()?;
        }
        Ok(())
    }

    fn exhausted(&self, spent: f64) -> BudgetError {
        let limit = self.inner.limit_usd();
        let id = self.scope_id.clone();
        match self.scope_kind {
            ScopeKind::Agent => BudgetError::AgentExhausted { agent_id: id, spent, limit },
            ScopeKind::Task => BudgetError::TaskExhausted { task_id: id, spent, limit },
            ScopeKind::Session => BudgetError::SessionExhausted { session_id: id, spent, limit },
        }
    }

    /// Activate the kill switch: every later charge is refused with `reason`.
    pub fn kill(&self, reason: &str) {
        *lock(&self.kill_reason) = Some(reason.to_string());
        self.killed.store(true, Ordering::SeqCst);
    }

    /// The reason given to [`kill`](Self::kill), if the switch is active.
    pub fn kill_reason(&self) -> Option<String> {
        lock(&self.kill_reason).clone()
    }

    /// True once [`kill`](Self::kill) has been called.
    pub fn is_killed(&self) -> bool {
        self.killed.load(Ordering::SeqCst)
    }

    /// Spent so far, in USD.
    pub fn spent_usd(&self) -> f64 {
        self.inner.spent_usd()
    }

    /// Held by in-flight reservations, in USD.
    pub fn reserved_usd(&self) -> f64 {
        self.inner.reserved_usd()
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

    /// Fraction of the cap spent (0.0 to 1.0, above 1.0 after a forced charge).
    pub fn utilization(&self) -> f64 {
        self.inner.utilization()
    }

    /// The records the log still holds, oldest first (see [`AuditLog`]).
    pub async fn audit_log(&self) -> Vec<SpendRecord> {
        lock(&self.log).records.iter().cloned().collect()
    }

    /// Take every record the log holds, oldest first, leaving it empty.
    pub fn drain_audit_log(&self) -> Vec<SpendRecord> {
        lock(&self.log).records.drain(..).collect()
    }

    /// How many records were evicted (or never kept) because of the
    /// [`AuditLog`] setting. Drained records do not count.
    pub fn audit_records_dropped(&self) -> u64 {
        lock(&self.log).dropped
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
    cascade: Arc<Mutex<ModelCascade>>,
    /// `None` means the built-in table.
    prices: Option<Arc<PriceTable>>,
    /// What to do when a model has no usable price; `None` refuses with
    /// `UnknownModel` (the behavior before 0.7).
    policy: Option<UnpricedPolicy>,
    /// 0 for a root budget, parent depth + 1 for a child scope.
    depth: u32,
    /// The deepest child scope this tree allows.
    max_depth: u32,
}

/// Builds an [`AgentBudget`]; see [`AgentBudget::builder`].
#[derive(Debug, Clone)]
pub struct AgentBudgetBuilder {
    agent_id: String,
    limit_usd: f64,
    tier: ModelTier,
    cascade_threshold: f64,
    audit_log: AuditLog,
    prices: Option<Arc<PriceTable>>,
    policy: Option<UnpricedPolicy>,
    max_depth: u32,
}

impl AgentBudgetBuilder {
    /// The tier the agent starts on (default [`ModelTier::Standard`]).
    pub fn tier(mut self, tier: ModelTier) -> Self {
        self.tier = tier;
        self
    }

    /// Step down one tier whenever the remaining fraction of the budget is at
    /// or below this (0.0 to 1.0, default 0.2).
    pub fn cascade_threshold(mut self, threshold: f64) -> Self {
        self.cascade_threshold = threshold;
        self
    }

    /// How much of the audit log to keep in memory (default `Ring(1024)`).
    pub fn audit_log(mut self, mode: AuditLog) -> Self {
        self.audit_log = mode;
        self
    }

    /// Price every model-based charge, [`record`](AgentBudget::record) and
    /// [`reserve_for`](AgentBudget::reserve_for) from `table` instead of
    /// [`PriceTable::builtin`]. This is the hook for negotiated rates,
    /// fine-tunes and self-hosted models; see the [`pricing`](crate::pricing)
    /// module docs. Pass an `Arc` to share one table between agents.
    ///
    /// ```
    /// use llm_budget::{AgentBudget, ModelPrice, PriceTable, Usage};
    /// # fn main() -> Result<(), llm_budget::BudgetError> {
    /// let table = PriceTable::builtin().clone().with_price(ModelPrice::new("my-llama", "self", 0.1, 0.1));
    /// let agent = AgentBudget::builder("a", 1.0).price_table(table).build()?;
    /// agent.record("my-llama", &Usage::new(1_000_000, 0))?;
    /// assert!((agent.spent_usd() - 0.1).abs() < 1e-9);
    /// # Ok(())
    /// # }
    /// ```
    pub fn price_table(mut self, table: impl Into<Arc<PriceTable>>) -> Self {
        self.prices = Some(table.into());
        self
    }

    /// What to do with a call whose model has no usable price, per
    /// [`PriceStatus`]; see [`UnpricedPolicy`].
    /// Child scopes inherit it. Without one, such calls are refused with
    /// [`BudgetError::UnknownModel`].
    pub fn unpriced_policy(mut self, policy: UnpricedPolicy) -> Self {
        self.policy = Some(policy);
        self
    }

    /// The deepest [child scope](AgentBudget::child) this budget's tree may
    /// have (the root is depth 0; default [`DEFAULT_MAX_DEPTH`]). A deeper
    /// [`child`](AgentBudget::child) fails with [`BudgetError::DepthExceeded`].
    pub fn max_depth(mut self, max_depth: u32) -> Self {
        self.max_depth = max_depth;
        self
    }

    /// Build the budget.
    ///
    /// # Errors
    /// [`BudgetError::InvalidConfig`] for a non-positive limit, a threshold
    /// outside 0..=1, or an [`UnpricedPolicy`] with a negative `Reserve`.
    pub fn build(self) -> Result<AgentBudget, BudgetError> {
        if let Some(p) = &self.policy {
            p.validate()?;
        }
        let envelope = BudgetEnvelope::new(self.agent_id.clone(), ScopeKind::Agent, self.limit_usd)?
            .with_audit_log(self.audit_log);
        let cascade = ModelCascade::new(self.tier, self.cascade_threshold, self.limit_usd)?;
        Ok(AgentBudget {
            agent_id: self.agent_id,
            envelope,
            cascade: Arc::new(Mutex::new(cascade)),
            prices: self.prices,
            policy: self.policy,
            depth: 0,
            max_depth: self.max_depth,
        })
    }
}

impl AgentBudget {
    /// Create an agent with a `limit_usd` cap, starting on `initial_tier`, that
    /// steps down one tier whenever the remaining fraction of its budget is at
    /// or below `cascade_threshold` (0.0 to 1.0). The audit log keeps the last
    /// 1,024 records; use [`builder`](Self::builder) to change that.
    ///
    /// # Errors
    /// [`BudgetError::InvalidConfig`] for a non-positive limit or a threshold outside 0..=1.
    pub fn new(
        agent_id: impl Into<String>,
        limit_usd: f64,
        initial_tier: ModelTier,
        cascade_threshold: f64,
    ) -> Result<Self, BudgetError> {
        Self::builder(agent_id, limit_usd)
            .tier(initial_tier)
            .cascade_threshold(cascade_threshold)
            .build()
    }

    /// Start building an agent budget with a `limit_usd` cap.
    ///
    /// ```
    /// use llm_budget::{AgentBudget, AuditLog, ModelTier};
    /// # fn main() -> Result<(), llm_budget::BudgetError> {
    /// let agent = AgentBudget::builder("indexer", 25.0)
    ///     .tier(ModelTier::Economy)
    ///     .audit_log(AuditLog::Ring(10_000))
    ///     .build()?;
    /// assert_eq!(agent.limit_usd(), 25.0);
    /// # Ok(())
    /// # }
    /// ```
    pub fn builder(agent_id: impl Into<String>, limit_usd: f64) -> AgentBudgetBuilder {
        AgentBudgetBuilder {
            agent_id: agent_id.into(),
            limit_usd,
            tier: ModelTier::Standard,
            cascade_threshold: 0.2,
            audit_log: AuditLog::default(),
            prices: None,
            policy: None,
            max_depth: DEFAULT_MAX_DEPTH,
        }
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
    /// agent's [`PriceTable`] (the built-in one unless set with
    /// [`AgentBudgetBuilder::price_table`]).
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
        let cost = self.price_table().cost(model, u64::from(input_tokens), u64::from(output_tokens))?;
        self.charge_inner(Some(model), Some(cost), input_tokens, output_tokens, task_id, session_id)
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
    /// agent's [`PriceTable`]. Refused, booking nothing, if it would cross the
    /// cap; for a call that already happened prefer [`record`](Self::record).
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
        let cost = self.price_table().cost_usage(model, usage)?;
        let clamp = |n: u64| u32::try_from(n).unwrap_or(u32::MAX);
        self.charge_inner(Some(model), Some(cost), clamp(usage.input_tokens), clamp(usage.output_tokens), task_id, session_id)
            .await
    }

    /// Charge an exact dollar amount (for example the cost your provider reported).
    ///
    /// The charge is refused, and nothing is booked, if it would cross the
    /// cap. For a call that has already been made, use
    /// [`charge_usd_force`](Self::charge_usd_force) instead so the spend is
    /// not lost.
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

    /// Book `cost_usd` of spend that has already happened, whatever the cap
    /// and the kill switch say, and write it to the audit log.
    ///
    /// Use this after a provider call you did not [`reserve`](Self::reserve)
    /// for: refusing it would only make [`spent_usd`](Self::spent_usd) lower
    /// than the bill. Check [`ChargeOutcome::over_cap`] to decide whether to
    /// keep going. Negative, NaN and infinite amounts book $0.
    ///
    /// On an agent from [`FleetAllocator::get_agent`](crate::FleetAllocator::get_agent)
    /// this books against the agent only: it is not added to
    /// [`FleetBudget::spent_usd`](crate::FleetBudget::spent_usd).
    pub fn charge_usd_force(&self, cost_usd: f64, ctx: ChargeContext) -> ChargeOutcome {
        settle(&self.agent_id, &self.envelope, &self.cascade, 0, cost_usd, ctx)
    }

    /// The table this agent prices models from: the built-in one unless set
    /// with [`AgentBudgetBuilder::price_table`].
    pub fn price_table(&self) -> &PriceTable {
        match &self.prices {
            Some(t) => t,
            None => PriceTable::builtin(),
        }
    }

    /// What `usage` on `model` costs at this agent's prices, in USD. Books
    /// nothing.
    ///
    /// # Errors
    /// [`BudgetError::UnknownModel`] if the model has no price.
    pub fn cost_of(&self, model: &str, usage: &crate::usage::Usage) -> Result<f64, BudgetError> {
        self.price_table().cost_usage(model, usage)
    }

    /// Book a call that has already happened, from the token counts the
    /// provider reported, whatever client or framework made it.
    ///
    /// This is the framework-agnostic way to report spend: make the call
    /// however you like, build a [`Usage`](crate::Usage) from its response
    /// ([`Usage::from_parts`](crate::Usage::from_parts),
    /// [`Usage::from_json`](crate::Usage::from_json), or a `From` impl), and
    /// record it. Like [`charge_usd_force`](Self::charge_usd_force) it always
    /// books, even past the cap, because the money is already spent;
    /// [`ChargeOutcome::over_cap`] says whether to stop. To also stop calls
    /// *before* they overspend, use [`reserve_for`](Self::reserve_for) and
    /// [`Reservation::settle_usage`].
    ///
    /// ```
    /// use llm_budget::{AgentBudget, Usage};
    /// # fn main() -> Result<(), llm_budget::BudgetError> {
    /// let agent = AgentBudget::builder("summariser", 1.00).build()?;
    /// // 12k-token prompt, 10k of it from the cache, 400 output tokens.
    /// let out = agent.record("gpt-4o", &Usage::new(12_000, 400).with_cache_read(10_240))?;
    /// assert!(!out.over_cap);
    /// assert!((agent.spent_usd() - out.recorded_usd).abs() < 1e-12);
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// With an [`UnpricedPolicy`], a model with no usable price is booked at
    /// $0 (`Allow`) or at the policy's flat amount (`Reserve`), or refused
    /// (`Deny`).
    ///
    /// # Errors
    /// [`BudgetError::UnknownModel`] (no policy) or [`BudgetError::Unpriced`]
    /// (policy `Deny`) if the model has no price; nothing is booked then (book
    /// a dollar amount with [`charge_usd_force`](Self::charge_usd_force)
    /// instead).
    pub fn record(&self, model: &str, usage: &crate::usage::Usage) -> Result<ChargeOutcome, BudgetError> {
        self.record_with(model, usage, ChargeContext::new())
    }

    /// [`record`](Self::record), with a [`ChargeContext`] for the audit log.
    /// The model and token counts are filled in from `model` and `usage`
    /// unless `ctx` already sets them.
    ///
    /// # Errors
    /// Same as [`record`](Self::record).
    pub fn record_with(
        &self,
        model: &str,
        usage: &crate::usage::Usage,
        ctx: ChargeContext,
    ) -> Result<ChargeOutcome, BudgetError> {
        let (cost, _) = self.price_or_policy(model, usage)?;
        Ok(self.charge_usd_force(cost, fill_ctx(ctx, model, usage)))
    }

    /// Why `model` can or cannot be priced for `usage` at this agent's
    /// prices (see [`PriceTable::lookup`]).
    pub fn price_status(&self, model: &str, usage: &crate::usage::Usage) -> PriceStatus {
        self.price_table().lookup(model, usage).status
    }

    /// The agent's unpriced-model policy, if it has one.
    pub fn unpriced_policy(&self) -> Option<&UnpricedPolicy> {
        self.policy.as_ref()
    }

    /// The cost of `usage` on `model`, or what the [`UnpricedPolicy`] says to
    /// hold or book when it has no usable price.
    fn price_or_policy(&self, model: &str, usage: &crate::usage::Usage) -> Result<(f64, PriceStatus), BudgetError> {
        let found = self.price_table().lookup(model, usage);
        if let Some(usd) = found.cost_usd {
            return Ok((usd, PriceStatus::Exact));
        }
        let status = found.status;
        let action = self.policy.as_ref().and_then(|p| p.action(status));
        match action {
            None => Err(BudgetError::UnknownModel(model.to_string())),
            Some(PriceAction::Allow) => Ok((0.0, status)),
            Some(PriceAction::Reserve(usd)) => Ok((usd, status)),
            Some(PriceAction::Deny) => {
                let e = BudgetError::Unpriced { model: model.to_string(), status };
                crate::observe::refused(&self.agent_id, false, &e.to_string());
                Err(e)
            }
        }
    }

    /// Hold headroom for a call to `model` whose expected size is `estimate`
    /// (prompt tokens plus the output you allow), priced from this agent's
    /// table. Settle it with [`Reservation::settle_usage`] once the provider
    /// reports the real usage.
    ///
    /// ```
    /// use llm_budget::{AgentBudget, Usage};
    /// # fn main() -> Result<(), llm_budget::BudgetError> {
    /// let agent = AgentBudget::builder("a", 0.10).build()?;
    /// let hold = agent.reserve_for("gpt-4o", &Usage::new(8_000, 1_000))?; // $0.03 held
    /// // ... make the call with any client ...
    /// let out = hold.settle_usage("gpt-4o", &Usage::new(7_412, 230));
    /// assert!(!out.over_cap);
    /// assert_eq!(agent.reserved_usd(), 0.0);
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// When the model has no usable price, the agent's [`UnpricedPolicy`]
    /// decides: hold nothing, hold a flat amount, or refuse
    /// ([`Reservation::price_status`] says which case applied).
    ///
    /// # Errors
    /// [`BudgetError::Unpriced`] when the policy denies an unpriced model,
    /// [`BudgetError::UnknownModel`] when the model has no price and the agent
    /// has no policy, otherwise the same as [`reserve`](Self::reserve).
    pub fn reserve_for(&self, model: &str, estimate: &crate::usage::Usage) -> Result<Reservation, BudgetError> {
        let (usd, status) = self.price_or_policy(model, estimate)?;
        let mut hold = self.reserve(usd)?;
        hold.model = Some(model.to_string());
        hold.status = status;
        Ok(hold)
    }

    /// Hold `estimate_usd` of headroom for a call you are about to make.
    ///
    /// The hold is taken with a compare-and-swap against the cap, so
    /// concurrent callers can never be admitted for more than the cap in
    /// total. Book the real cost with [`Reservation::settle`]; dropping the
    /// reservation without settling releases the hold.
    ///
    /// # Errors
    /// [`BudgetError::InvalidConfig`] for a negative or non-finite estimate,
    /// [`BudgetError::KillSwitchActivated`] after [`kill`](Self::kill), or
    /// [`BudgetError::AgentExhausted`] when the estimate does not fit in what
    /// is left.
    ///
    /// On an agent from [`FleetAllocator::get_agent`](crate::FleetAllocator::get_agent)
    /// the settled cost is booked against the agent only: it is not added to
    /// [`FleetBudget::spent_usd`](crate::FleetBudget::spent_usd).
    pub fn reserve(&self, estimate_usd: f64) -> Result<Reservation, BudgetError> {
        if !estimate_usd.is_finite() || estimate_usd < 0.0 {
            return Err(BudgetError::InvalidConfig(format!("estimate must be >= 0, got {estimate_usd}")));
        }
        self.envelope.check_alive()?;
        let nanos = usd_to_nanos(estimate_usd);
        if let Err(spent) = self.envelope.atomic().reserve_nanos(nanos) {
            let e = self.envelope.exhausted(spent);
            crate::observe::refused(&self.agent_id, false, &e.to_string());
            return Err(e);
        }
        Ok(Reservation {
            agent_id: self.agent_id.clone(),
            envelope: self.envelope.clone(),
            cascade: Arc::clone(&self.cascade),
            prices: self.prices.clone(),
            model: None,
            status: PriceStatus::Exact,
            nanos,
            open: true,
        })
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
        let current_tier = lock(&self.cascade).current();
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
        let downgraded_to = lock(&self.cascade).evaluate(remaining);
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

    /// Settled spend so far, in USD (in-flight reservations not included).
    pub fn spent_usd(&self) -> f64 {
        self.envelope.spent_usd()
    }

    /// Held by in-flight reservations, in USD.
    pub fn reserved_usd(&self) -> f64 {
        self.envelope.reserved_usd()
    }

    /// Left to spend or reserve, in USD.
    pub fn remaining_usd(&self) -> f64 {
        self.envelope.remaining_usd()
    }

    /// The agent's cap, in USD.
    pub fn limit_usd(&self) -> f64 {
        self.envelope.limit_usd()
    }

    /// True when nothing is left to spend or reserve.
    pub fn is_exhausted(&self) -> bool {
        self.envelope.is_exhausted()
    }

    /// Fraction of the cap spent or held by open reservations and child
    /// scopes (0.0 to 1.0; above 1.0 only after a settlement larger than its
    /// reservation or a forced charge).
    pub fn utilization(&self) -> f64 {
        self.envelope.utilization()
    }

    /// The tier the agent should use for its next call.
    pub async fn current_tier(&self) -> ModelTier {
        lock(&self.cascade).current()
    }

    /// The records the audit log still holds, oldest first (see [`AuditLog`]).
    pub async fn audit_log(&self) -> Vec<SpendRecord> {
        self.envelope.audit_log().await
    }

    /// Take every record the audit log holds, oldest first, leaving it empty.
    /// Call it periodically to ship records to your own storage.
    pub fn drain_audit_log(&self) -> Vec<SpendRecord> {
        self.envelope.drain_audit_log()
    }

    /// How many audit records were evicted or never kept because of the
    /// [`AuditLog`] setting.
    pub fn audit_records_dropped(&self) -> u64 {
        self.envelope.audit_records_dropped()
    }

    /// Move `amount_usd` of unspent cap out of this agent. Returns the new cap.
    pub(crate) async fn shrink_limit(&self, amount_usd: f64) -> Result<f64, f64> {
        let new_limit = self.envelope.atomic().shrink_limit(amount_usd)?;
        lock(&self.cascade).set_total_budget(new_limit);
        Ok(new_limit)
    }

    /// Add `amount_usd` to this agent's cap. Returns the new cap.
    pub(crate) async fn grow_limit(&self, amount_usd: f64) -> f64 {
        let new_limit = self.envelope.atomic().grow_limit(amount_usd);
        lock(&self.cascade).set_total_budget(new_limit);
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
            model_tier: lock(&self.cascade).current(),
            timestamp_ms: now_ms(),
        }
    }
}

/// Release `reserved` nanos and book `cost_usd` in their place, with a log
/// record, observability events and a cascade check.
fn settle(
    agent_id: &str,
    envelope: &BudgetEnvelope,
    cascade: &Mutex<ModelCascade>,
    reserved: u64,
    cost_usd: f64,
    ctx: ChargeContext,
) -> ChargeOutcome {
    let current_tier = lock(cascade).current();
    let cost = nanos_to_usd(usd_to_nanos(sanitize_usd(cost_usd)));
    let model = ctx.model;
    let record = SpendRecord {
        agent_id: agent_id.to_string(),
        task_id: ctx.task_id,
        session_id: ctx.session_id,
        model_tier: current_tier,
        model: model.clone(),
        input_tokens: ctx.input_tokens,
        output_tokens: ctx.output_tokens,
        cost_usd: cost,
        timestamp_ms: now_ms(),
    };
    let (total, over_cap) = envelope.book(record, reserved);
    let remaining = envelope.remaining_usd();
    crate::observe::charged(agent_id, model.as_deref(), cost, total, remaining);
    let downgraded_to = lock(cascade).evaluate(remaining);
    if let Some(tier) = downgraded_to {
        crate::observe::downgraded(agent_id, tier);
    }
    ChargeOutcome { recorded_usd: cost, total_spent_usd: total, over_cap, downgraded_to }
}

/// Headroom held for one in-flight call; see [`AgentBudget::reserve`].
///
/// Settle it with the real cost once the call returns. If it is dropped
/// unsettled (the call failed, the task was cancelled) the hold is released
/// and nothing is booked.
#[must_use = "dropping a Reservation releases it; call settle() with the real cost"]
pub struct Reservation {
    agent_id: String,
    envelope: BudgetEnvelope,
    cascade: Arc<Mutex<ModelCascade>>,
    prices: Option<Arc<PriceTable>>,
    /// The model given to `reserve_for`, a fallback for pricing the settlement.
    model: Option<String>,
    /// How the estimate was priced.
    status: PriceStatus,
    nanos: u64,
    open: bool,
}

impl Reservation {
    /// The amount held, in USD.
    pub fn estimate_usd(&self) -> f64 {
        nanos_to_usd(self.nanos)
    }

    /// The model this hold was taken for with
    /// [`reserve_for`](AgentBudget::reserve_for), if any.
    pub fn model(&self) -> Option<&str> {
        self.model.as_deref()
    }

    /// How the held amount was priced: [`PriceStatus::Exact`] for a priced
    /// model (and for [`reserve`](AgentBudget::reserve), where you priced it),
    /// otherwise the status the [`UnpricedPolicy`] acted on.
    pub fn price_status(&self) -> PriceStatus {
        self.status
    }

    /// Release the hold and book `actual_usd`, even if it is more than the
    /// estimate and takes the agent past its cap. Negative, NaN and infinite
    /// amounts book $0.
    pub fn settle(self, actual_usd: f64) -> ChargeOutcome {
        self.settle_with(actual_usd, ChargeContext::default())
    }

    /// [`settle`](Self::settle), recording `ctx` in the audit log.
    pub fn settle_with(mut self, actual_usd: f64, ctx: ChargeContext) -> ChargeOutcome {
        self.open = false;
        settle(&self.agent_id, &self.envelope, &self.cascade, self.nanos, actual_usd, ctx)
    }

    /// Release the hold and book what `usage` on `model` costs at the
    /// agent's prices. This is the settlement half of the
    /// [`reserve_for`](AgentBudget::reserve_for) / `settle_usage` pattern and
    /// works with any client: build the [`Usage`](crate::Usage) from whatever
    /// the provider returned.
    ///
    /// If `model` has no price (a dated id the provider echoes back, say), the
    /// model given to [`reserve_for`](AgentBudget::reserve_for) is tried, and
    /// failing that the reserved estimate is booked (the call happened, so
    /// leaving it off the books would undercount). Check with
    /// [`AgentBudget::cost_of`] first if you need to know. Like
    /// [`settle`](Self::settle), this never fails and books past the cap.
    pub fn settle_usage(self, model: &str, usage: &crate::usage::Usage) -> ChargeOutcome {
        self.settle_usage_with(model, usage, ChargeContext::new())
    }

    /// [`settle_usage`](Self::settle_usage) with a [`ChargeContext`]; the
    /// model and token counts are filled in unless `ctx` sets them.
    pub fn settle_usage_with(self, model: &str, usage: &crate::usage::Usage, ctx: ChargeContext) -> ChargeOutcome {
        let table = match &self.prices {
            Some(t) => t.as_ref(),
            None => PriceTable::builtin(),
        };
        let priced = table.cost_usage(model, usage).map(|usd| (usd, model)).or_else(|e| match self.model.as_deref() {
            Some(m) => table.cost_usage(m, usage).map(|usd| (usd, m)),
            None => Err(e),
        });
        let (usd, model) = match priced {
            Ok((usd, m)) => (usd, m.to_string()),
            Err(_) => (self.estimate_usd(), model.to_string()),
        };
        self.settle_with(usd, fill_ctx(ctx, &model, usage))
    }

    /// Release the hold without booking anything (same as dropping it).
    pub fn release(self) {}
}

/// Fill the model and token counts of `ctx` from a usage report, keeping any
/// the caller set.
fn fill_ctx(mut ctx: ChargeContext, model: &str, usage: &crate::usage::Usage) -> ChargeContext {
    if ctx.model.is_none() {
        ctx.model = Some(model.to_string());
    }
    if ctx.input_tokens == 0 && ctx.output_tokens == 0 {
        ctx = ctx.usage(usage);
    }
    ctx
}

impl Drop for Reservation {
    fn drop(&mut self) {
        if self.open {
            self.envelope.release(self.nanos);
        }
    }
}

impl std::fmt::Debug for Reservation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Reservation")
            .field("agent_id", &self.agent_id)
            .field("estimate_usd", &self.estimate_usd())
            .field("model", &self.model)
            .field("open", &self.open)
            .finish()
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
        #[cfg(feature = "store")]
        check::<crate::BudgetStore>();
        check::<Reservation>();
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
        #[cfg(feature = "store")]
        {
            let snap = serde_json::to_value(schemars::schema_for!(crate::persistence::PersistedBudgetState)).unwrap();
            assert!(snap["properties"]["snapshots"].is_object());
        }
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
    // ── 0.3.0: settlement, reservations, bounded audit log ─────────────────

    #[tokio::test]
    async fn test_force_charge_books_past_the_cap_and_the_kill_switch() {
        let agent = AgentBudget::new("a", 1.0, ModelTier::Standard, 0.1).expect("new");
        agent.charge_usd(0.9, None, None).await.expect("fits");
        assert!(agent.charge_usd(0.5, None, None).await.is_err());
        let out = agent.charge_usd_force(0.5, ChargeContext::new().task("t"));
        assert!(out.over_cap);
        assert!((out.recorded_usd - 0.5).abs() < 1e-12);
        assert!((agent.spent_usd() - 1.4).abs() < 1e-9);
        agent.kill("stop");
        let out = agent.charge_usd_force(0.1, ChargeContext::default());
        assert!((out.total_spent_usd - 1.5).abs() < 1e-9);
        let log = agent.audit_log().await;
        assert_eq!(log.len(), 3);
        assert_eq!(log[1].task_id.as_deref(), Some("t"));
        let sum: f64 = log.iter().map(|r| r.cost_usd).sum();
        assert!((sum - agent.spent_usd()).abs() < 1e-9);
    }

    #[test]
    fn test_force_charge_ignores_bad_amounts() {
        let agent = AgentBudget::new("a", 1.0, ModelTier::Standard, 0.1).expect("new");
        for x in [f64::NAN, -1.0, f64::INFINITY, f64::NEG_INFINITY] {
            assert_eq!(agent.charge_usd_force(x, ChargeContext::new()).recorded_usd, 0.0);
        }
        assert_eq!(agent.spent_usd(), 0.0);
    }

    #[test]
    fn test_reservation_holds_headroom_and_drop_releases_it() {
        let agent = AgentBudget::new("a", 1.0, ModelTier::Standard, 0.1).expect("new");
        let r = agent.reserve(0.75).expect("fits");
        assert!((agent.reserved_usd() - 0.75).abs() < 1e-12);
        assert_eq!(agent.spent_usd(), 0.0);
        assert!(agent.reserve(0.5).is_err(), "only 0.25 is left");
        drop(r);
        assert_eq!(agent.reserved_usd(), 0.0);
        assert!((agent.remaining_usd() - 1.0).abs() < 1e-12);
        let r = agent.reserve(0.5).expect("fits again");
        r.release();
        assert!((agent.remaining_usd() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_settle_books_the_actual_cost() {
        let agent = AgentBudget::new("a", 1.0, ModelTier::Standard, 0.1).expect("new");
        let out = agent.reserve(0.2).expect("fits").settle(0.05);
        assert!(!out.over_cap);
        assert!((agent.spent_usd() - 0.05).abs() < 1e-12);
        assert_eq!(agent.reserved_usd(), 0.0);
        // a call that ran far over its estimate is still booked in full
        let out = agent.reserve(0.1).expect("fits").settle(2.0);
        assert!(out.over_cap);
        assert!((agent.spent_usd() - 2.05).abs() < 1e-9);
        assert!(agent.is_exhausted());
    }

    #[test]
    fn test_reserve_refuses_bad_estimates_and_killed_budgets() {
        let agent = AgentBudget::new("a", 1.0, ModelTier::Standard, 0.1).expect("new");
        assert!(matches!(agent.reserve(-0.1), Err(BudgetError::InvalidConfig(_))));
        assert!(matches!(agent.reserve(f64::NAN), Err(BudgetError::InvalidConfig(_))));
        assert!(matches!(agent.reserve(2.0), Err(BudgetError::AgentExhausted { .. })));
        agent.kill("x");
        assert!(matches!(agent.reserve(0.1), Err(BudgetError::KillSwitchActivated { .. })));
    }

    #[tokio::test]
    async fn test_audit_log_ring_evicts_oldest() {
        let agent = AgentBudget::builder("a", 100.0).audit_log(AuditLog::Ring(3)).build().expect("new");
        for i in 0..5u32 {
            agent.charge_usd_force(f64::from(i), ChargeContext::new());
        }
        let log = agent.audit_log().await;
        assert_eq!(log.iter().map(|r| r.cost_usd).collect::<Vec<_>>(), vec![2.0, 3.0, 4.0]);
        assert_eq!(agent.audit_records_dropped(), 2);
        assert_eq!(agent.drain_audit_log().len(), 3);
        assert!(agent.audit_log().await.is_empty());
    }

    #[tokio::test]
    async fn test_audit_log_default_is_bounded_and_none_keeps_nothing() {
        let agent = AgentBudget::new("a", 1e6, ModelTier::Local, 0.1).expect("new");
        for _ in 0..1_500 {
            agent.charge_usd(0.001, None, None).await.expect("ok");
        }
        assert_eq!(agent.audit_log().await.len(), 1024);
        assert_eq!(agent.audit_records_dropped(), 476);

        let quiet = AgentBudget::builder("q", 10.0).audit_log(AuditLog::None).build().expect("new");
        quiet.charge_usd(1.0, None, None).await.expect("ok");
        assert!(quiet.audit_log().await.is_empty());
        assert!((quiet.spent_usd() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_builder_validates() {
        assert!(AgentBudget::builder("a", 0.0).build().is_err());
        assert!(AgentBudget::builder("a", 1.0).cascade_threshold(1.5).build().is_err());
    }

    #[test]
    fn test_reservation_is_send_and_sync() {
        fn check<T: Send + Sync + 'static>() {}
        check::<Reservation>();
        check::<ChargeOutcome>();
    }
}
