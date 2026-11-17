//! Child budget scopes: reserve a slice of a parent budget for a sub-agent,
//! settle what it really spent, refund the rest.
//!
//! # Why
//!
//! An agent that delegates to several sub-agents at once cannot hand each of
//! them "what is left": every child checks against the same remainder, none
//! of them sees its siblings, and together they spend a multiple of it before
//! any of them reports back. A child scope fixes that by moving from
//! *check, then spend* to *reserve, then settle* at the delegation boundary:
//!
//! 1. [`AgentBudget::child`] reserves the child's whole limit from the parent
//!    with a compare-and-swap, before the child starts. Siblings racing for
//!    the last dollar cannot all be admitted.
//! 2. The child is an ordinary [`AgentBudget`] (through [`ScopeGuard`]'s
//!    `Deref`): `reserve_for`, `record`, `charge_*`, and further `child`
//!    scopes all work on it. Every amount it books is booked on each ancestor
//!    at the same moment, inside the hold, so a parent's
//!    [`spent_usd`](AgentBudget::spent_usd) is always current.
//! 3. [`ScopeGuard::settle`] (or dropping the guard, also on a panic or a
//!    cancelled task) closes the child and gives the unused part of the hold
//!    back to the parent.
//!
//! The invariant, under any interleaving: a parent's open child holds plus
//! everything booked on it never exceed its limit through admission. The
//! only way past a limit is a settlement larger than its reservation (a call
//! that cost more than estimated, or `charge_usd_force`): that money is
//! already spent, so it is booked in full, up every level, and
//! [`ChargeOutcome::over_cap`](crate::ChargeOutcome::over_cap) reports it.
//!
//! ```
//! use llm_budget::{AgentBudget, BudgetError, Limit};
//!
//! # fn main() -> Result<(), BudgetError> {
//! let root = AgentBudget::builder("orchestrator", 1.00).build()?;
//!
//! let a = root.child("researcher", Limit::Usd(0.60))?;
//! let b = root.child("writer", Limit::Fraction(0.5))?; // half of the $0.40 left
//! assert!((root.remaining_usd() - 0.20).abs() < 1e-9);
//!
//! // A third child asking for more than is left is refused up front.
//! assert!(root.child("critic", Limit::Usd(0.50)).is_err());
//!
//! a.reserve(0.10)?.settle(0.07); // the researcher's call cost $0.07
//! assert!((root.spent_usd() - 0.07).abs() < 1e-9); // visible on the parent at once
//!
//! let receipt = a.settle(); // close it: $0.53 of its hold goes back
//! assert!((receipt.released_usd - 0.53).abs() < 1e-9);
//! drop(b); // dropping releases the writer's untouched $0.20 too
//! assert!((root.remaining_usd() - 0.93).abs() < 1e-9);
//! # Ok(())
//! # }
//! ```
//!
//! # Fan-out on tokio
//!
//! [`ScopeGuard`] is `Send + Sync + 'static`, so it moves into a spawned task.
//! See `examples/fanout.rs` for four parallel sub-agents under one cap.
//!
//! # Closing
//!
//! A closed scope admits nothing new: `reserve`, `charge_*` and `child`
//! return [`BudgetError::ScopeClosed`], for it and for every scope below it.
//! Calls already in flight still settle and are booked; the part of the hold
//! they need stays reserved in the parent until they do. Killing a scope
//! ([`AgentBudget::kill`]) stops it and every scope below it.

use std::ops::Deref;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use super::{
    lock, nanos_to_usd, usd_to_nanos, AgentBudget, AtomicBudget, BudgetEnvelope, ScopeKind, SpendRecord,
};
use crate::error::BudgetError;
use crate::model::ModelCascade;

/// The deepest child scope a budget tree allows unless set with
/// [`AgentBudgetBuilder::max_depth`](crate::AgentBudgetBuilder::max_depth).
pub const DEFAULT_MAX_DEPTH: u32 = 8;

/// How much of its parent a [child scope](AgentBudget::child) reserves.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Limit {
    /// A fixed amount in USD (finite and above zero).
    Usd(f64),
    /// A fraction (above 0, at most 1) of what the parent has left to spend
    /// or reserve when the child is created.
    Fraction(f64),
}

/// The link from a child scope's envelope to the hold it draws on.
pub(crate) struct ParentLink {
    parent: BudgetEnvelope,
    /// The part of the child's hold in the parent not yet converted into
    /// booked spend or released, in nano-dollars.
    hold: AtomicU64,
}

impl ParentLink {
    pub(crate) fn parent(&self) -> &BudgetEnvelope {
        &self.parent
    }

    /// Book `actual` nanos the child just spent on the parent, out of the
    /// hold where it still covers them.
    pub(crate) fn forward(&self, record: &SpendRecord, actual: u64) {
        let take = take_up_to(&self.hold, actual);
        self.parent.book_from_child(record, take, actual);
    }

    /// Release every part of the hold the closed child's in-flight
    /// reservations do not need. Returns the nanos released.
    pub(crate) fn trim(&self, child: &AtomicBudget) -> u64 {
        let keep = child.reserved_nanos();
        let mut current = self.hold.load(Ordering::SeqCst);
        loop {
            if current <= keep {
                return 0;
            }
            match self.hold.compare_exchange_weak(current, keep, Ordering::SeqCst, Ordering::SeqCst) {
                Ok(_) => {
                    let released = current - keep;
                    self.parent.release(released);
                    return released;
                }
                Err(seen) => current = seen,
            }
        }
    }
}

/// Subtract up to `want` from `a`; returns what was subtracted.
fn take_up_to(a: &AtomicU64, want: u64) -> u64 {
    let mut current = a.load(Ordering::SeqCst);
    loop {
        let take = current.min(want);
        if take == 0 {
            return 0;
        }
        match a.compare_exchange_weak(current, current - take, Ordering::SeqCst, Ordering::SeqCst) {
            Ok(_) => return take,
            Err(seen) => current = seen,
        }
    }
}

/// A child budget scope; see [`AgentBudget::child`] and the
/// [module docs](self).
///
/// It dereferences to the child's [`AgentBudget`], so every budget method
/// works on it, including [`child`](AgentBudget::child) for a grandchild.
/// [`settle`](Self::settle) closes it and returns a [`ScopeReceipt`];
/// dropping it closes it the same way.
///
/// `Send + Sync + 'static`: move it into a spawned task, or share it with
/// [`budget`](Self::budget) (an `Arc<AgentBudget>`, for integrations such as
/// `rig::BudgetHook` that take one). Clones of that `Arc` stay usable until
/// the guard closes the scope; after that they refuse new calls with
/// [`BudgetError::ScopeClosed`].
#[must_use = "dropping a ScopeGuard closes the scope and refunds its hold at once"]
pub struct ScopeGuard {
    budget: Arc<AgentBudget>,
    held_nanos: u64,
    open: bool,
}

/// What a child scope did, returned by [`ScopeGuard::settle`].
///
/// `own_cost_usd + child_cost_usd == spent_usd`: the split follows the A2A
/// budget extension draft (a2aproject/A2A#2121), where an agent reports its
/// own spend apart from what its delegates spent.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[non_exhaustive]
pub struct ScopeReceipt {
    /// The scope's name.
    pub scope_id: String,
    /// Its depth (the root is 0).
    pub depth: u32,
    /// What it reserved from its parent, in USD.
    pub held_usd: f64,
    /// Everything booked on it, its own calls and its descendants', in USD.
    pub spent_usd: f64,
    /// The part of `spent_usd` from its own calls.
    pub own_cost_usd: f64,
    /// The part of `spent_usd` forwarded from its descendant scopes.
    pub child_cost_usd: f64,
    /// Hold given back to the parent when it closed, in USD.
    pub released_usd: f64,
    /// Hold still reserved in the parent for calls in flight when it
    /// closed, in USD. It is booked or released as those calls settle.
    pub in_flight_usd: f64,
    /// True when it booked more than it held (a settlement above its
    /// estimate, or a forced charge).
    pub over_hold: bool,
}

impl AgentBudget {
    /// Carve a child scope out of this budget: reserve `limit` from it now,
    /// for a sub-agent, a delegated task or one branch of a fan-out.
    ///
    /// The child is a full [`AgentBudget`] with a cap of the reserved amount,
    /// this budget's price table, [`UnpricedPolicy`](crate::UnpricedPolicy),
    /// audit-log setting, current tier and cascade threshold. Everything it
    /// books is booked on this budget at the same time (and on every
    /// ancestor). Closing it ([`ScopeGuard::settle`] or drop) refunds the
    /// unused part. See the [module docs](crate::budget::scope).
    ///
    /// ```
    /// use llm_budget::{AgentBudget, Limit};
    /// # fn main() -> Result<(), llm_budget::BudgetError> {
    /// let parent = AgentBudget::builder("planner", 2.00).build()?;
    /// let child = parent.child("search", Limit::Usd(0.50))?;
    /// let grandchild = child.child("fetch", Limit::Fraction(0.2))?; // $0.10
    /// assert_eq!(grandchild.depth(), 2);
    /// grandchild.reserve(0.04)?.settle(0.04);
    /// assert!((parent.spent_usd() - 0.04).abs() < 1e-9);
    /// assert!((child.child_cost_usd() - 0.04).abs() < 1e-9);
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Errors
    /// [`BudgetError::InvalidConfig`] for a non-finite or non-positive
    /// amount or a fraction outside (0, 1]; [`BudgetError::DepthExceeded`]
    /// past [`max_depth`](Self::max_depth); [`BudgetError::ScopeClosed`] or
    /// [`BudgetError::KillSwitchActivated`] when this scope (or an ancestor)
    /// is closed or killed; [`BudgetError::AgentExhausted`] when the amount
    /// does not fit in what this budget has left.
    pub fn child(&self, scope_id: impl Into<String>, limit: Limit) -> Result<ScopeGuard, BudgetError> {
        let scope_id = scope_id.into();
        self.envelope.check_alive()?;
        let depth = self.depth.saturating_add(1);
        if depth > self.max_depth {
            return Err(BudgetError::DepthExceeded { scope_id, depth, max_depth: self.max_depth });
        }
        let atomic = self.envelope.atomic();
        let nanos = match limit {
            Limit::Usd(usd) if usd.is_finite() && usd > 0.0 => usd_to_nanos(usd),
            Limit::Fraction(f) if f.is_finite() && f > 0.0 && f <= 1.0 => {
                let remaining = atomic.limit_nanos().saturating_sub(atomic.committed_nanos());
                (remaining as f64 * f).floor() as u64
            }
            Limit::Usd(x) => return Err(BudgetError::InvalidConfig(format!("Limit::Usd must be > 0, got {x}"))),
            Limit::Fraction(x) => {
                return Err(BudgetError::InvalidConfig(format!("Limit::Fraction must be in (0, 1], got {x}")))
            }
        };
        if nanos == 0 {
            return Err(self.envelope.exhausted(self.spent_usd()));
        }
        let cascade = {
            let c = lock(&self.cascade);
            ModelCascade::new(c.current(), c.threshold(), nanos_to_usd(nanos))?
        };
        if let Err(spent) = atomic.reserve_nanos(nanos) {
            let e = self.envelope.exhausted(spent);
            crate::observe::refused(&self.agent_id, false, &e.to_string());
            return Err(e);
        }
        self.envelope.delegations.fetch_add(1, Ordering::SeqCst);
        let audit = lock(&self.envelope.log).mode;
        let link = Arc::new(ParentLink { parent: self.envelope.clone(), hold: AtomicU64::new(nanos) });
        let envelope = BudgetEnvelope::with_atomic(
            scope_id.clone(),
            ScopeKind::Agent,
            AtomicBudget::from_nanos(nanos),
            audit,
            Some(link),
        );
        let budget = AgentBudget {
            agent_id: scope_id,
            envelope,
            cascade: Arc::new(Mutex::new(cascade)),
            prices: self.prices.clone(),
            policy: self.policy,
            depth,
            max_depth: self.max_depth,
        };
        Ok(ScopeGuard { budget: Arc::new(budget), held_nanos: nanos, open: true })
    }

    /// 0 for a root budget, one more than its parent for a child scope.
    pub fn depth(&self) -> u32 {
        self.depth
    }

    /// The deepest child scope this budget's tree allows.
    pub fn max_depth(&self) -> u32 {
        self.max_depth
    }

    /// How many child scopes were created directly under this one (the A2A
    /// draft's `incurred.delegations`).
    pub fn delegations(&self) -> u64 {
        self.envelope.delegations.load(Ordering::SeqCst)
    }

    /// What is left to spend or reserve, in nano-dollars.
    pub(crate) fn remaining_nanos(&self) -> u64 {
        let a = self.envelope.atomic();
        a.limit_nanos().saturating_sub(a.committed_nanos())
    }

    /// [`own_cost_usd`](Self::own_cost_usd) in nano-dollars.
    pub(crate) fn own_cost_nanos(&self) -> u64 {
        let a = self.envelope.atomic();
        a.committed_nanos()
            .saturating_sub(a.reserved_nanos())
            .saturating_sub(self.envelope.child_nanos.load(Ordering::SeqCst))
    }

    /// [`child_cost_usd`](Self::child_cost_usd) in nano-dollars.
    pub(crate) fn child_cost_nanos(&self) -> u64 {
        self.envelope.child_nanos.load(Ordering::SeqCst)
    }

    /// The parent scope's name, for a child scope.
    pub fn parent_id(&self) -> Option<&str> {
        self.envelope.parent.as_ref().map(|l| l.parent().scope_id.as_str())
    }

    /// The part of [`spent_usd`](Self::spent_usd) forwarded from descendant
    /// scopes (the A2A draft's `aggregatedChildCost`).
    pub fn child_cost_usd(&self) -> f64 {
        nanos_to_usd(self.envelope.child_nanos.load(Ordering::SeqCst))
    }

    /// The part of [`spent_usd`](Self::spent_usd) booked by this scope's own
    /// calls, not its descendants'.
    pub fn own_cost_usd(&self) -> f64 {
        nanos_to_usd(self.own_cost_nanos())
    }

    /// True once this child scope was settled or dropped.
    pub fn is_closed(&self) -> bool {
        self.envelope.closed.load(Ordering::SeqCst)
    }
}

impl ScopeGuard {
    /// The child budget as an `Arc`, for integrations that take one.
    pub fn budget(&self) -> &Arc<AgentBudget> {
        &self.budget
    }

    /// What the scope reserved from its parent, in USD.
    pub fn held_usd(&self) -> f64 {
        nanos_to_usd(self.held_nanos)
    }

    /// [`held_usd`](Self::held_usd) in nano-dollars.
    pub(crate) fn held_nanos(&self) -> u64 {
        self.held_nanos
    }

    /// Close the scope: it admits nothing new, and the part of its hold that
    /// no call in flight needs goes back to the parent at once.
    pub fn settle(mut self) -> ScopeReceipt {
        self.close()
    }

    fn close(&mut self) -> ScopeReceipt {
        self.open = false;
        let env = &self.budget.envelope;
        env.closed.store(true, Ordering::SeqCst);
        let released = env.parent.as_ref().map_or(0, |link| link.trim(env.atomic()));
        let child = env.child_nanos.load(Ordering::SeqCst);
        let atomic = env.atomic();
        let spent = atomic.committed_nanos().saturating_sub(atomic.reserved_nanos());
        let in_flight = env.parent.as_ref().map_or(0, |l| l.hold.load(Ordering::SeqCst));
        ScopeReceipt {
            scope_id: env.scope_id.clone(),
            depth: self.budget.depth,
            held_usd: nanos_to_usd(self.held_nanos),
            spent_usd: nanos_to_usd(spent),
            own_cost_usd: nanos_to_usd(spent.saturating_sub(child)),
            child_cost_usd: nanos_to_usd(child),
            released_usd: nanos_to_usd(released),
            in_flight_usd: nanos_to_usd(in_flight),
            over_hold: spent > self.held_nanos,
        }
    }
}

impl Deref for ScopeGuard {
    type Target = AgentBudget;
    fn deref(&self) -> &AgentBudget {
        &self.budget
    }
}

impl Drop for ScopeGuard {
    fn drop(&mut self) {
        if self.open {
            let _ = self.close();
        }
    }
}

impl std::fmt::Debug for ScopeGuard {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ScopeGuard")
            .field("scope_id", &self.budget.agent_id)
            .field("parent_id", &self.budget.parent_id())
            .field("depth", &self.budget.depth)
            .field("held_usd", &self.held_usd())
            .field("spent_usd", &self.budget.spent_usd())
            .field("open", &self.open)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ChargeContext, Usage};

    fn root(limit: f64) -> AgentBudget {
        AgentBudget::builder("root", limit).build().expect("root")
    }

    fn near(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9
    }

    #[test]
    fn guard_is_send_sync_static() {
        fn assert_traits<T: Send + Sync + 'static>() {}
        assert_traits::<ScopeGuard>();
        assert_traits::<ScopeReceipt>();
        assert_traits::<AgentBudget>();
    }

    #[test]
    fn child_reserves_and_refunds() {
        let r = root(1.0);
        let c = r.child("c", Limit::Usd(0.4)).expect("child");
        assert!(near(r.remaining_usd(), 0.6));
        assert!(near(r.reserved_usd(), 0.4));
        assert!(near(r.spent_usd(), 0.0));
        assert!(near(c.limit_usd(), 0.4));
        c.reserve(0.1).expect("hold").settle(0.08);
        assert!(near(r.spent_usd(), 0.08));
        assert!(near(r.reserved_usd(), 0.32));
        let rec = c.settle();
        assert!(near(rec.spent_usd, 0.08));
        assert!(near(rec.released_usd, 0.32));
        assert!(near(rec.own_cost_usd, 0.08));
        assert!(near(rec.child_cost_usd, 0.0));
        assert!(!rec.over_hold);
        assert!(near(r.remaining_usd(), 0.92));
        assert!(near(r.reserved_usd(), 0.0));
        assert!(near(r.child_cost_usd(), 0.08));
        assert!(near(r.own_cost_usd(), 0.0));
    }

    #[test]
    fn fraction_is_of_what_is_left() {
        let r = root(1.0);
        r.charge_usd_force(0.2, ChargeContext::new());
        let c = r.child("c", Limit::Fraction(0.5)).expect("child");
        assert!(near(c.limit_usd(), 0.4));
        let d = r.child("d", Limit::Fraction(1.0)).expect("rest");
        assert!(near(d.limit_usd(), 0.4));
        assert!(matches!(r.child("e", Limit::Fraction(0.5)), Err(BudgetError::AgentExhausted { .. })));
    }

    #[test]
    fn bad_limits_are_refused() {
        let r = root(1.0);
        for l in [Limit::Usd(0.0), Limit::Usd(-1.0), Limit::Usd(f64::NAN), Limit::Fraction(0.0), Limit::Fraction(1.5)] {
            assert!(matches!(r.child("x", l), Err(BudgetError::InvalidConfig(_))), "{l:?}");
        }
        assert!(matches!(r.child("big", Limit::Usd(1.01)), Err(BudgetError::AgentExhausted { .. })));
        assert!(near(r.remaining_usd(), 1.0));
    }

    #[test]
    fn depth_limit() {
        let r = AgentBudget::builder("root", 1.0).max_depth(2).build().expect("root");
        let a = r.child("a", Limit::Fraction(0.5)).expect("a");
        let b = a.child("b", Limit::Fraction(0.5)).expect("b");
        let e = b.child("c", Limit::Fraction(0.5)).expect_err("too deep");
        assert_eq!(e, BudgetError::DepthExceeded { scope_id: "c".into(), depth: 3, max_depth: 2 });
        assert_eq!(b.depth(), 2);
        assert_eq!(b.parent_id(), Some("a"));
        // the refused child reserved nothing
        assert!(near(b.remaining_usd(), 0.25));
    }

    #[test]
    fn closed_scope_refuses_new_work_but_settles_in_flight() {
        let r = root(1.0);
        let c = r.child("c", Limit::Usd(0.5)).expect("child");
        let shared = Arc::clone(c.budget());
        let hold = c.reserve(0.2).expect("hold");
        let rec = c.settle();
        assert!(near(rec.in_flight_usd, 0.2));
        assert!(near(rec.released_usd, 0.3));
        assert!(matches!(shared.reserve(0.01), Err(BudgetError::ScopeClosed { .. })));
        assert!(matches!(shared.child("g", Limit::Usd(0.01)), Err(BudgetError::ScopeClosed { .. })));
        assert!(near(r.reserved_usd(), 0.2));
        hold.settle(0.15);
        assert!(near(r.spent_usd(), 0.15));
        assert!(near(r.reserved_usd(), 0.0));
        assert!(near(r.remaining_usd(), 0.85));
    }

    #[test]
    fn dropped_in_flight_reservation_after_close_is_refunded() {
        let r = root(1.0);
        let c = r.child("c", Limit::Usd(0.5)).expect("child");
        let hold = c.reserve(0.2).expect("hold");
        drop(c);
        assert!(near(r.reserved_usd(), 0.2));
        drop(hold);
        assert!(near(r.reserved_usd(), 0.0));
        assert!(near(r.remaining_usd(), 1.0));
    }

    #[test]
    fn overrun_is_booked_up_every_level() {
        let r = root(1.0);
        let a = r.child("a", Limit::Usd(0.5)).expect("a");
        let b = a.child("b", Limit::Usd(0.2)).expect("b");
        let out = b.reserve(0.2).expect("hold").settle(0.35);
        assert!(out.over_cap);
        assert!(near(b.spent_usd(), 0.35));
        assert!(near(a.spent_usd(), 0.35));
        assert!(near(r.spent_usd(), 0.35));
        let rb = b.settle();
        assert!(rb.over_hold);
        let ra = a.settle();
        assert!(near(ra.child_cost_usd, 0.35));
        assert!(near(ra.own_cost_usd, 0.0));
        // a held 0.5 and booked 0.35 from b: 0.15 back
        assert!(near(ra.released_usd, 0.15));
        assert!(near(r.remaining_usd(), 0.65));
    }

    #[test]
    fn grandchild_spend_is_split_into_own_and_child_cost() {
        let r = root(2.0);
        let a = r.child("a", Limit::Usd(1.0)).expect("a");
        a.charge_usd_force(0.1, ChargeContext::new());
        let b = a.child("b", Limit::Usd(0.5)).expect("b");
        b.charge_usd_force(0.2, ChargeContext::new());
        let c = b.child("c", Limit::Usd(0.1)).expect("c");
        c.charge_usd_force(0.05, ChargeContext::new());
        assert!(near(a.own_cost_usd(), 0.1));
        assert!(near(a.child_cost_usd(), 0.25));
        assert!(near(b.own_cost_usd(), 0.2));
        assert!(near(b.child_cost_usd(), 0.05));
        assert!(near(r.spent_usd(), 0.35));
        assert!(near(r.child_cost_usd(), 0.35));
    }

    #[test]
    fn killing_an_ancestor_stops_descendants() {
        let r = root(1.0);
        let a = r.child("a", Limit::Usd(0.5)).expect("a");
        let b = a.child("b", Limit::Usd(0.2)).expect("b");
        r.kill("operator stop");
        let e = b.reserve(0.01).expect_err("killed");
        assert!(matches!(e, BudgetError::KillSwitchActivated { ref agent_id, .. } if agent_id == "root"));
        // spend that already happened is still booked
        assert!(!b.charge_usd_force(0.01, ChargeContext::new()).over_cap);
        assert!(near(r.spent_usd(), 0.01));
    }

    #[test]
    fn audit_log_sums_match_at_every_level() {
        let r = AgentBudget::builder("root", 5.0).audit_log(crate::AuditLog::Unbounded).build().expect("root");
        let a = r.child("a", Limit::Usd(2.0)).expect("a");
        let b = a.child("b", Limit::Usd(1.0)).expect("b");
        b.record("gpt-4o", &Usage::new(10_000, 500)).expect("priced");
        a.reserve(0.1).expect("hold").settle(0.05);
        r.charge_usd_force(0.5, ChargeContext::new());
        let sum = |log: Vec<SpendRecord>| log.iter().map(|x| x.cost_usd).sum::<f64>();
        let rt = tokio_free_block(r.audit_log());
        let at = tokio_free_block(a.audit_log());
        assert!(near(sum(rt.clone()), r.spent_usd()));
        assert!(near(sum(at), a.spent_usd()));
        assert!(rt.iter().any(|x| x.agent_id == "b"));
    }

    /// The audit_log futures never await anything, so a no-op waker drives them.
    fn tokio_free_block<F: std::future::Future>(f: F) -> F::Output {
        use std::task::{Context, Poll, Waker};
        let mut f = std::pin::pin!(f);
        let mut cx = Context::from_waker(Waker::noop());
        loop {
            if let Poll::Ready(v) = f.as_mut().poll(&mut cx) {
                return v;
            }
        }
    }

    #[test]
    fn policy_and_prices_are_inherited() {
        use crate::pricing::{PriceAction, PriceStatus, UnpricedPolicy};
        let policy = UnpricedPolicy::new(PriceAction::Allow, PriceAction::Deny, PriceAction::Deny);
        let table = crate::PriceTable::builtin().clone().with_unpriced("local-llama");
        let r = AgentBudget::builder("root", 1.0).price_table(table).unpriced_policy(policy).build().expect("root");
        let c = r.child("c", Limit::Usd(0.5)).expect("c");
        let h = c.reserve_for("local-llama", &Usage::new(100, 100)).expect("allowed");
        assert_eq!(h.price_status(), PriceStatus::Unpriced);
        assert!(near(h.estimate_usd(), 0.0));
        h.release();
        assert!(matches!(
            c.reserve_for("nope-model-z", &Usage::new(1, 1)),
            Err(BudgetError::Unpriced { status: PriceStatus::Missing, .. })
        ));
    }
}
