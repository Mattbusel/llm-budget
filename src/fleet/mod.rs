//! Fleet-wide budget allocation and reallocation across agent cohorts.

use crate::budget::{nanos_to_usd, usd_to_nanos, AgentBudget, BudgetSnapshot, ChargeReceipt};
use crate::error::BudgetError;
use crate::model::ModelTier;
use dashmap::DashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// Fleet-level budget: the total that may be handed out to agents, and what they spent.
pub struct FleetBudget {
    /// Total fleet budget in nano-dollars.
    fleet_limit_nanos: u64,
    /// Allocated nano-dollars (committed to agents).
    allocated_nanos: AtomicU64,
    /// Spent nano-dollars (actually consumed).
    spent_nanos: AtomicU64,
}

impl FleetBudget {
    /// Create a fleet with a `limit_usd` total.
    ///
    /// # Errors
    /// [`BudgetError::InvalidConfig`] if the limit is not positive.
    pub fn new(limit_usd: f64) -> Result<Self, BudgetError> {
        if !limit_usd.is_finite() || limit_usd <= 0.0 {
            return Err(BudgetError::InvalidConfig("fleet limit must be positive".into()));
        }
        Ok(Self {
            fleet_limit_nanos: usd_to_nanos(limit_usd),
            allocated_nanos: AtomicU64::new(0),
            spent_nanos: AtomicU64::new(0),
        })
    }

    /// Atomically allocate `amount_usd` to an agent. Returns available balance after.
    ///
    /// # Errors
    /// [`BudgetError::AllocationFailed`] if the fleet does not have that much unallocated.
    pub fn allocate(&self, agent_id: &str, amount_usd: f64) -> Result<f64, BudgetError> {
        let amount_nanos = usd_to_nanos(amount_usd);
        let limit = self.fleet_limit_nanos;
        // `fetch_update` is deprecated in favour of `try_update` on Rust 1.99+,
        // which is newer than this crate's minimum, so keep it for now.
        #[allow(deprecated)]
        match self.allocated_nanos.fetch_update(Ordering::SeqCst, Ordering::SeqCst, |prev| {
            prev.checked_add(amount_nanos).filter(|&n| n <= limit)
        }) {
            Ok(prev) => Ok(nanos_to_usd(limit - (prev + amount_nanos))),
            Err(prev) => Err(BudgetError::AllocationFailed {
                agent_id: agent_id.to_string(),
                requested: amount_usd,
                available: nanos_to_usd(limit.saturating_sub(prev)),
            }),
        }
    }

    /// Give `amount_usd` of allocation back to the fleet.
    fn release(&self, amount_usd: f64) {
        let amount = usd_to_nanos(amount_usd);
        #[allow(deprecated)] // see `allocate`: try_update needs Rust 1.99
        let _ = self
            .allocated_nanos
            .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |prev| Some(prev.saturating_sub(amount)));
    }

    /// Record actual spend (called by [`FleetAllocator`] after each successful charge).
    pub fn record_spend(&self, amount_usd: f64) {
        self.spent_nanos.fetch_add(usd_to_nanos(amount_usd), Ordering::SeqCst);
    }

    /// The fleet total, in USD.
    pub fn total_limit_usd(&self) -> f64 { nanos_to_usd(self.fleet_limit_nanos) }
    /// Sum of all agent caps handed out, in USD.
    pub fn allocated_usd(&self) -> f64 { nanos_to_usd(self.allocated_nanos.load(Ordering::SeqCst)) }
    /// Sum of what all agents actually spent, in USD.
    pub fn spent_usd(&self) -> f64 { nanos_to_usd(self.spent_nanos.load(Ordering::SeqCst)) }
    /// Not yet handed out to any agent, in USD.
    pub fn available_usd(&self) -> f64 {
        let alloc = self.allocated_nanos.load(Ordering::SeqCst);
        nanos_to_usd(self.fleet_limit_nanos.saturating_sub(alloc))
    }
    /// True when the whole fleet total has been handed out.
    pub fn is_exhausted(&self) -> bool {
        self.allocated_nanos.load(Ordering::SeqCst) >= self.fleet_limit_nanos
    }
}

/// Fleet allocator: the registry of agents and how the fleet budget is split between them.
pub struct FleetAllocator {
    fleet: Arc<FleetBudget>,
    agents: DashMap<String, Arc<AgentBudget>>,
    default_tier: ModelTier,
    default_cascade_threshold: f64,
}

impl FleetAllocator {
    /// Create a fleet with a `fleet_limit_usd` total. New agents start on
    /// `default_tier` and downgrade at `default_cascade_threshold`.
    ///
    /// # Errors
    /// [`BudgetError::InvalidConfig`] if the limit is not positive.
    pub fn new(
        fleet_limit_usd: f64,
        default_tier: ModelTier,
        default_cascade_threshold: f64,
    ) -> Result<Self, BudgetError> {
        Ok(Self {
            fleet: Arc::new(FleetBudget::new(fleet_limit_usd)?),
            agents: DashMap::new(),
            default_tier,
            default_cascade_threshold,
        })
    }

    /// Register a new agent with the given per-agent limit.
    ///
    /// # Errors
    /// [`BudgetError::AllocationFailed`] if the fleet cannot cover the limit,
    /// [`BudgetError::InvalidConfig`] if the id is already registered or the limit is invalid.
    pub fn register_agent(&self, agent_id: &str, limit_usd: f64) -> Result<(), BudgetError> {
        use dashmap::mapref::entry::Entry;
        let budget = AgentBudget::new(
            agent_id,
            limit_usd,
            self.default_tier,
            self.default_cascade_threshold,
        )?;
        match self.agents.entry(agent_id.to_string()) {
            Entry::Occupied(_) => Err(BudgetError::InvalidConfig(format!(
                "agent '{agent_id}' is already registered"
            ))),
            Entry::Vacant(slot) => {
                self.fleet.allocate(agent_id, limit_usd)?;
                slot.insert(Arc::new(budget));
                Ok(())
            }
        }
    }

    /// Remove an agent and return its unspent cap to the fleet.
    ///
    /// # Errors
    /// [`BudgetError::UnknownAgent`] if no such agent is registered.
    pub fn remove_agent(&self, agent_id: &str) -> Result<RemovedAgent, BudgetError> {
        let (_, agent) = self
            .agents
            .remove(agent_id)
            .ok_or_else(|| BudgetError::UnknownAgent(agent_id.to_string()))?;
        // The agent's spent money stays counted against the fleet allocation;
        // only the unspent part is released.
        let remaining = agent.remaining_usd();
        self.fleet.release(remaining);
        Ok(RemovedAgent { spent_usd: agent.spent_usd(), released_usd: remaining })
    }

    fn agent(&self, agent_id: &str) -> Result<Arc<AgentBudget>, BudgetError> {
        self.agents
            .get(agent_id)
            .map(|a| Arc::clone(a.value()))
            .ok_or_else(|| BudgetError::UnknownAgent(agent_id.to_string()))
    }

    /// Charge an agent by ID, priced by its current tier. Returns
    /// `(agent_total_spent, Some(tier))` when the agent was just downgraded.
    ///
    /// # Errors
    /// [`BudgetError::UnknownAgent`], or any error from [`AgentBudget::charge`].
    pub async fn charge_agent(
        &self,
        agent_id: &str,
        input_tokens: u32,
        output_tokens: u32,
        task_id: Option<String>,
        session_id: Option<String>,
    ) -> Result<(f64, Option<ModelTier>), BudgetError> {
        let agent = self.agent(agent_id)?;
        let r = agent.charge_inner(None, None, input_tokens, output_tokens, task_id, session_id).await?;
        self.fleet.record_spend(r.cost_usd);
        Ok((r.total_spent_usd, r.downgraded_to))
    }

    /// Charge an agent for a call to a named model at that model's real price
    /// (see [`crate::pricing::PriceTable`]).
    ///
    /// # Errors
    /// [`BudgetError::UnknownAgent`], [`BudgetError::UnknownModel`], or any
    /// error from [`AgentBudget::charge_model`].
    pub async fn charge_agent_model(
        &self,
        agent_id: &str,
        model: &str,
        input_tokens: u32,
        output_tokens: u32,
        task_id: Option<String>,
        session_id: Option<String>,
    ) -> Result<ChargeReceipt, BudgetError> {
        let agent = self.agent(agent_id)?;
        let r = agent.charge_model(model, input_tokens, output_tokens, task_id, session_id).await?;
        self.fleet.record_spend(r.cost_usd);
        Ok(r)
    }

    /// Charge an agent from a provider-reported [`crate::usage::Usage`]
    /// (see [`AgentBudget::charge_usage`]).
    ///
    /// # Errors
    /// [`BudgetError::UnknownAgent`], [`BudgetError::UnknownModel`], or any
    /// error from the charge.
    pub async fn charge_agent_usage(
        &self,
        agent_id: &str,
        model: &str,
        usage: &crate::usage::Usage,
        task_id: Option<String>,
        session_id: Option<String>,
    ) -> Result<ChargeReceipt, BudgetError> {
        let agent = self.agent(agent_id)?;
        let r = agent.charge_usage(model, usage, task_id, session_id).await?;
        self.fleet.record_spend(r.cost_usd);
        Ok(r)
    }

    /// The budget of one registered agent. Charges made directly on it are not
    /// added to [`FleetBudget::spent_usd`]; go through the fleet for that.
    pub fn get_agent(&self, agent_id: &str) -> Option<Arc<AgentBudget>> {
        self.agent(agent_id).ok()
    }

    /// Kill a specific agent's budget.
    ///
    /// # Errors
    /// [`BudgetError::UnknownAgent`] if no such agent is registered.
    pub fn kill_agent(&self, agent_id: &str, reason: &str) -> Result<(), BudgetError> {
        self.agent(agent_id)?.kill(reason);
        Ok(())
    }

    /// Kill all agents (emergency fleet shutdown).
    pub fn kill_all(&self, reason: &str) {
        for agent in self.agents.iter() {
            agent.kill(reason);
        }
    }

    /// Move `amount_usd` of unspent cap from one agent to another. The fleet
    /// total is unchanged.
    ///
    /// # Errors
    /// [`BudgetError::UnknownAgent`] if either agent is missing,
    /// [`BudgetError::InvalidConfig`] for a non-positive amount or the same agent twice,
    /// [`BudgetError::ReallocationFailed`] if the source has less than `amount_usd` left.
    pub async fn reallocate(
        &self,
        from_agent: &str,
        to_agent: &str,
        amount_usd: f64,
    ) -> Result<(), BudgetError> {
        if !amount_usd.is_finite() || amount_usd <= 0.0 {
            return Err(BudgetError::InvalidConfig("reallocation amount must be positive".into()));
        }
        if from_agent == to_agent {
            return Err(BudgetError::InvalidConfig("cannot reallocate an agent to itself".into()));
        }
        let from = self.agent(from_agent)?;
        let to = self.agent(to_agent)?;
        from.shrink_limit(amount_usd).await.map_err(|available| BudgetError::ReallocationFailed {
            from: from_agent.to_string(),
            to: to_agent.to_string(),
            amount: amount_usd,
            available,
        })?;
        to.grow_limit(amount_usd).await;
        Ok(())
    }

    /// Real-time spend snapshot for all agents.
    pub async fn fleet_snapshot(&self) -> Vec<BudgetSnapshot> {
        let mut snapshots = Vec::with_capacity(self.agents.len());
        for entry in self.agents.iter() {
            snapshots.push(entry.snapshot().await);
        }
        snapshots
    }

    /// The fleet-level numbers.
    pub fn fleet_budget(&self) -> &FleetBudget {
        &self.fleet
    }

    /// Number of registered agents.
    pub fn agent_count(&self) -> usize {
        self.agents.len()
    }

    /// True when the whole fleet total has been handed out to agents.
    pub fn is_fleet_exhausted(&self) -> bool {
        self.fleet.is_exhausted()
    }
}

/// What [`FleetAllocator::remove_agent`] did.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
pub struct RemovedAgent {
    /// What the removed agent had spent, in USD (stays counted against the fleet).
    pub spent_usd: f64,
    /// Unspent cap returned to the fleet, in USD.
    pub released_usd: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_fleet(limit: f64) -> FleetAllocator {
        FleetAllocator::new(limit, ModelTier::Standard, 0.2).expect("fleet")
    }

    #[test]
    fn test_fleet_budget_rejects_zero_limit() {
        assert!(FleetBudget::new(0.0).is_err());
        assert!(FleetBudget::new(-10.0).is_err());
    }

    #[test]
    fn test_fleet_budget_allocate_within_limit() {
        let fleet = FleetBudget::new(100.0).expect("new");
        let result = fleet.allocate("a1", 30.0);
        assert!(result.is_ok());
        assert!((fleet.available_usd() - 70.0).abs() < 1e-4);
    }

    #[test]
    fn test_fleet_budget_allocate_over_limit_rejected() {
        let fleet = FleetBudget::new(10.0).expect("new");
        fleet.allocate("a1", 8.0).expect("first");
        let result = fleet.allocate("a2", 5.0);
        assert!(matches!(result, Err(BudgetError::AllocationFailed { .. })));
        // Available should still reflect only 8.0 allocated
        assert!((fleet.available_usd() - 2.0).abs() < 1e-4);
    }

    #[test]
    fn test_fleet_budget_exhausted_when_fully_allocated() {
        let fleet = FleetBudget::new(10.0).expect("new");
        fleet.allocate("a1", 10.0).expect("alloc");
        assert!(fleet.is_exhausted());
    }

    #[test]
    fn test_fleet_allocator_register_agent() {
        let fleet = make_fleet(100.0);
        fleet.register_agent("a1", 50.0).expect("register");
        assert_eq!(fleet.agent_count(), 1);
    }

    #[test]
    fn test_fleet_allocator_register_over_budget_fails() {
        let fleet = make_fleet(10.0);
        fleet.register_agent("a1", 8.0).expect("ok");
        let result = fleet.register_agent("a2", 5.0);
        assert!(matches!(result, Err(BudgetError::AllocationFailed { .. })));
    }

    #[tokio::test]
    async fn test_fleet_allocator_charge_unknown_agent_fails() {
        let fleet = make_fleet(100.0);
        let result = fleet.charge_agent("ghost", 100, 50, None, None).await;
        assert!(matches!(result, Err(BudgetError::UnknownAgent(_))));
    }

    #[tokio::test]
    async fn test_fleet_allocator_charge_registered_agent() {
        let fleet = make_fleet(100.0);
        fleet.register_agent("a1", 10.0).expect("register");
        let result = fleet.charge_agent("a1", 0, 0, None, None).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_fleet_kill_all_blocks_all_agents() {
        let fleet = make_fleet(100.0);
        fleet.register_agent("a1", 10.0).expect("reg");
        fleet.register_agent("a2", 10.0).expect("reg");
        fleet.kill_all("emergency shutdown");
        let r1 = fleet.charge_agent("a1", 100, 50, None, None).await;
        let r2 = fleet.charge_agent("a2", 100, 50, None, None).await;
        assert!(matches!(r1, Err(BudgetError::KillSwitchActivated { .. })));
        assert!(matches!(r2, Err(BudgetError::KillSwitchActivated { .. })));
    }

    #[tokio::test]
    async fn test_fleet_snapshot_contains_all_agents() {
        let fleet = make_fleet(100.0);
        fleet.register_agent("a1", 20.0).expect("reg");
        fleet.register_agent("a2", 20.0).expect("reg");
        let snapshots = fleet.fleet_snapshot().await;
        assert_eq!(snapshots.len(), 2);
    }

    #[tokio::test]
    async fn test_reallocate_fails_for_unknown_dest() {
        let fleet = make_fleet(100.0);
        fleet.register_agent("a1", 20.0).expect("reg");
        let result = fleet.reallocate("a1", "ghost", 5.0).await;
        assert!(matches!(result, Err(BudgetError::UnknownAgent(_))));
    }

    #[tokio::test]
    async fn test_kill_specific_agent() {
        let fleet = make_fleet(100.0);
        fleet.register_agent("a1", 10.0).expect("reg");
        fleet.kill_agent("a1", "test kill").expect("kill");
        let result = fleet.charge_agent("a1", 100, 50, None, None).await;
        assert!(matches!(result, Err(BudgetError::KillSwitchActivated { .. })));
    }

    #[test]
    fn test_kill_unknown_agent_returns_error() {
        let fleet = make_fleet(100.0);
        let result = fleet.kill_agent("ghost", "test");
        assert!(matches!(result, Err(BudgetError::UnknownAgent(_))));
    }

    #[test]
    fn test_fleet_allocator_new_invalid_config() {
        let result = FleetAllocator::new(0.0, ModelTier::Standard, 0.2);
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_fleet_spend_is_sum_of_charges() {
        // Before 0.2.0 charge_agent added the agent's running total to the fleet
        // spend after every charge, so three $3 charges showed as $18 fleet spend.
        let fleet = make_fleet(100.0);
        fleet.register_agent("a1", 50.0).expect("reg");
        for _ in 0..3 {
            // Standard tier: 1M input tokens = $3
            fleet.charge_agent("a1", 1_000_000, 0, None, None).await.expect("charge");
        }
        let spent = fleet.fleet_budget().spent_usd();
        assert!((spent - 9.0).abs() < 1e-6, "fleet spend {spent}, expected 9.0");
        fleet.charge_agent_model("a1", "gpt-4o", 1_000_000, 0, None, None).await.expect("model");
        assert!((fleet.fleet_budget().spent_usd() - 11.5).abs() < 1e-6);
    }

    #[tokio::test]
    async fn test_reallocate_moves_budget() {
        // Before 0.2.0 reallocate validated the request and then did nothing.
        let fleet = make_fleet(100.0);
        fleet.register_agent("a1", 20.0).expect("reg");
        fleet.register_agent("a2", 20.0).expect("reg");
        fleet.reallocate("a1", "a2", 5.0).await.expect("move");
        let snaps: std::collections::HashMap<_, _> =
            fleet.fleet_snapshot().await.into_iter().map(|s| (s.agent_id.clone(), s)).collect();
        assert!((snaps["a1"].limit_usd - 15.0).abs() < 1e-9);
        assert!((snaps["a2"].limit_usd - 25.0).abs() < 1e-9);
        assert!((fleet.fleet_budget().allocated_usd() - 40.0).abs() < 1e-9);
        // a1 now has only $15 to give.
        assert!(matches!(
            fleet.reallocate("a1", "a2", 16.0).await,
            Err(BudgetError::ReallocationFailed { .. })
        ));
        assert!(fleet.reallocate("a1", "a1", 1.0).await.is_err());
        assert!(fleet.reallocate("a1", "a2", -1.0).await.is_err());
    }

    #[test]
    fn test_register_duplicate_agent_fails_without_leaking_allocation() {
        // Before 0.2.0 registering the same id twice silently replaced the agent
        // (resetting its spend) and allocated its limit a second time.
        let fleet = make_fleet(100.0);
        fleet.register_agent("a1", 30.0).expect("reg");
        assert!(fleet.register_agent("a1", 30.0).is_err());
        assert!((fleet.fleet_budget().allocated_usd() - 30.0).abs() < 1e-9);
        assert_eq!(fleet.agent_count(), 1);
    }

    #[tokio::test]
    async fn test_remove_agent_releases_unspent_cap() {
        let fleet = make_fleet(100.0);
        fleet.register_agent("a1", 30.0).expect("reg");
        fleet.charge_agent("a1", 1_000_000, 0, None, None).await.expect("charge $3");
        let r = fleet.remove_agent("a1").expect("remove");
        assert!((r.released_usd - 27.0).abs() < 1e-9);
        assert!((fleet.fleet_budget().allocated_usd() - 3.0).abs() < 1e-9);
        assert!(fleet.remove_agent("a1").is_err());
    }
}
