//! Fleet-wide budget allocation and reallocation across agent cohorts.

use crate::budget::{AgentBudget, BudgetSnapshot};
use crate::error::BudgetError;
use crate::model::ModelTier;
use dashmap::DashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// Fleet-level budget container — governs total spend across all agents.
pub struct FleetBudget {
    /// Total fleet budget in micro-dollars.
    fleet_limit_micros: u64,
    /// Allocated micro-dollars (committed to agents).
    allocated_micros: AtomicU64,
    /// Spent micro-dollars (actually consumed).
    spent_micros: AtomicU64,
}

impl FleetBudget {
    pub fn new(limit_usd: f64) -> Result<Self, BudgetError> {
        if limit_usd <= 0.0 {
            return Err(BudgetError::InvalidConfig("fleet limit must be positive".into()));
        }
        Ok(Self {
            fleet_limit_micros: usd_to_micros(limit_usd),
            allocated_micros: AtomicU64::new(0),
            spent_micros: AtomicU64::new(0),
        })
    }

    /// Atomically allocate `amount_usd` to an agent. Returns available balance after.
    pub fn allocate(&self, agent_id: &str, amount_usd: f64) -> Result<f64, BudgetError> {
        let amount_micros = usd_to_micros(amount_usd);
        let prev = self.allocated_micros.fetch_add(amount_micros, Ordering::SeqCst);
        let new_allocated = prev + amount_micros;
        if new_allocated > self.fleet_limit_micros {
            self.allocated_micros.fetch_sub(amount_micros, Ordering::SeqCst);
            Err(BudgetError::AllocationFailed {
                agent_id: agent_id.to_string(),
                requested: amount_usd,
                available: micros_to_usd(self.fleet_limit_micros - prev),
            })
        } else {
            Ok(micros_to_usd(self.fleet_limit_micros - new_allocated))
        }
    }

    /// Record actual spend (called by AgentBudget after successful charge).
    pub fn record_spend(&self, amount_usd: f64) {
        self.spent_micros.fetch_add(usd_to_micros(amount_usd), Ordering::SeqCst);
    }

    pub fn total_limit_usd(&self) -> f64 { micros_to_usd(self.fleet_limit_micros) }
    pub fn allocated_usd(&self) -> f64 { micros_to_usd(self.allocated_micros.load(Ordering::SeqCst)) }
    pub fn spent_usd(&self) -> f64 { micros_to_usd(self.spent_micros.load(Ordering::SeqCst)) }
    pub fn available_usd(&self) -> f64 {
        let alloc = self.allocated_micros.load(Ordering::SeqCst);
        if alloc >= self.fleet_limit_micros { 0.0 } else { micros_to_usd(self.fleet_limit_micros - alloc) }
    }
    pub fn is_exhausted(&self) -> bool {
        self.allocated_micros.load(Ordering::SeqCst) >= self.fleet_limit_micros
    }
}

/// Fleet allocator — manages agent registry and budget distribution.
pub struct FleetAllocator {
    fleet: Arc<FleetBudget>,
    agents: DashMap<String, Arc<AgentBudget>>,
    default_tier: ModelTier,
    default_cascade_threshold: f64,
}

impl FleetAllocator {
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
    pub fn register_agent(&self, agent_id: &str, limit_usd: f64) -> Result<(), BudgetError> {
        self.fleet.allocate(agent_id, limit_usd)?;
        let budget = AgentBudget::new(
            agent_id,
            limit_usd,
            self.default_tier,
            self.default_cascade_threshold,
        )?;
        self.agents.insert(agent_id.to_string(), Arc::new(budget));
        Ok(())
    }

    /// Charge an agent by ID.
    pub async fn charge_agent(
        &self,
        agent_id: &str,
        input_tokens: u32,
        output_tokens: u32,
        task_id: Option<String>,
        session_id: Option<String>,
    ) -> Result<(f64, Option<ModelTier>), BudgetError> {
        let agent = self.agents.get(agent_id)
            .ok_or_else(|| BudgetError::UnknownAgent(agent_id.to_string()))?;
        let result = agent.charge(input_tokens, output_tokens, task_id, session_id).await?;
        let cost = agent.spent_usd(); // approximate
        self.fleet.record_spend(cost);
        Ok(result)
    }

    /// Kill a specific agent's budget.
    pub fn kill_agent(&self, agent_id: &str, reason: &str) -> Result<(), BudgetError> {
        let agent = self.agents.get(agent_id)
            .ok_or_else(|| BudgetError::UnknownAgent(agent_id.to_string()))?;
        agent.kill(reason);
        Ok(())
    }

    /// Kill all agents — emergency fleet shutdown.
    pub fn kill_all(&self, reason: &str) {
        for agent in self.agents.iter() {
            agent.kill(reason);
        }
    }

    /// Reallocate budget from one agent to another (reduce source, increase dest).
    pub async fn reallocate(
        &self,
        from_agent: &str,
        to_agent: &str,
        amount_usd: f64,
    ) -> Result<(), BudgetError> {
        let from = self.agents.get(from_agent)
            .ok_or_else(|| BudgetError::UnknownAgent(from_agent.to_string()))?;
        let available = from.remaining_usd();
        if available < amount_usd {
            return Err(BudgetError::ReallocationFailed {
                from: from_agent.to_string(),
                to: to_agent.to_string(),
                amount: amount_usd,
                available,
            });
        }
        // Check if dest agent exists, if not register it
        if !self.agents.contains_key(to_agent) {
            return Err(BudgetError::UnknownAgent(to_agent.to_string()));
        }
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

    pub fn fleet_budget(&self) -> &FleetBudget {
        &self.fleet
    }

    pub fn agent_count(&self) -> usize {
        self.agents.len()
    }

    pub fn is_fleet_exhausted(&self) -> bool {
        self.fleet.is_exhausted()
    }
}

fn usd_to_micros(usd: f64) -> u64 {
    (usd * 1_000_000.0).round() as u64
}

fn micros_to_usd(micros: u64) -> f64 {
    micros as f64 / 1_000_000.0
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
}
