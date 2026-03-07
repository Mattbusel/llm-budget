//! Typed error hierarchy for llm-budget. Every budget operation surfaces
//! a named, matchable error variant — no silent failures.

use crate::model::ModelTier;

#[derive(Debug, thiserror::Error, Clone, PartialEq)]
pub enum BudgetError {
    /// The agent has exhausted its per-agent spend limit.
    #[error("Agent '{agent_id}' budget exhausted: spent ${spent:.6}, limit ${limit:.6}")]
    AgentExhausted { agent_id: String, spent: f64, limit: f64 },

    /// A task-scoped budget envelope has been depleted.
    #[error("Task '{task_id}' budget exhausted: spent ${spent:.6}, limit ${limit:.6}")]
    TaskExhausted { task_id: String, spent: f64, limit: f64 },

    /// A session-scoped budget envelope has been depleted.
    #[error("Session '{session_id}' budget exhausted: spent ${spent:.6}, limit ${limit:.6}")]
    SessionExhausted { session_id: String, spent: f64, limit: f64 },

    /// The fleet-wide budget cap has been reached.
    #[error("Fleet budget exhausted: spent ${spent:.6}, limit ${limit:.6}")]
    FleetExhausted { spent: f64, limit: f64 },

    /// The kill switch was triggered; execution must halt.
    #[error("Kill switch activated for agent '{agent_id}': {reason}")]
    KillSwitchActivated { agent_id: String, reason: String },

    /// A model downgrade was required due to budget pressure.
    #[error("Model downgraded from {from:?} to {to:?} due to budget pressure (remaining: ${remaining:.6})")]
    ModelDowngraded { from: ModelTier, to: ModelTier, remaining: f64 },

    /// No lower-cost model is available for downgrade.
    #[error("No fallback model available from {from:?}; all tiers exhausted")]
    NoFallbackAvailable { from: ModelTier },

    /// Budget allocation failed — requested amount exceeds available fleet budget.
    #[error("Cannot allocate ${requested:.6} to agent '{agent_id}': only ${available:.6} available in fleet")]
    AllocationFailed { agent_id: String, requested: f64, available: f64 },

    /// Reallocation failed because the source agent has insufficient budget.
    #[error("Cannot reallocate ${amount:.6} from '{from}' to '{to}': source has ${available:.6}")]
    ReallocationFailed { from: String, to: String, amount: f64, available: f64 },

    /// Agent ID not found in the fleet registry.
    #[error("Agent '{0}' not registered in fleet")]
    UnknownAgent(String),

    /// Persistence I/O failure.
    #[error("Budget persistence error: {0}")]
    PersistenceError(String),

    /// Invalid configuration (e.g. negative limit).
    #[error("Invalid budget configuration: {0}")]
    InvalidConfig(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_agent_exhausted_display_contains_agent_id() {
        let e = BudgetError::AgentExhausted {
            agent_id: "agent-42".into(),
            spent: 1.5,
            limit: 1.0,
        };
        assert!(e.to_string().contains("agent-42"));
        assert!(e.to_string().contains("1.5"));
    }

    #[test]
    fn test_task_exhausted_display() {
        let e = BudgetError::TaskExhausted {
            task_id: "task-7".into(),
            spent: 0.05,
            limit: 0.04,
        };
        assert!(e.to_string().contains("task-7"));
    }

    #[test]
    fn test_session_exhausted_display() {
        let e = BudgetError::SessionExhausted {
            session_id: "sess-1".into(),
            spent: 2.0,
            limit: 1.0,
        };
        assert!(e.to_string().contains("sess-1"));
    }

    #[test]
    fn test_fleet_exhausted_display() {
        let e = BudgetError::FleetExhausted { spent: 100.0, limit: 99.0 };
        assert!(e.to_string().contains("Fleet"));
    }

    #[test]
    fn test_kill_switch_display() {
        let e = BudgetError::KillSwitchActivated {
            agent_id: "a1".into(),
            reason: "hard limit breach".into(),
        };
        assert!(e.to_string().contains("a1"));
        assert!(e.to_string().contains("hard limit breach"));
    }

    #[test]
    fn test_model_downgraded_display() {
        let e = BudgetError::ModelDowngraded {
            from: ModelTier::Premium,
            to: ModelTier::Standard,
            remaining: 0.01,
        };
        assert!(e.to_string().contains("downgraded"));
    }

    #[test]
    fn test_no_fallback_display() {
        let e = BudgetError::NoFallbackAvailable { from: ModelTier::Economy };
        assert!(e.to_string().contains("Economy"));
    }

    #[test]
    fn test_allocation_failed_display() {
        let e = BudgetError::AllocationFailed {
            agent_id: "a1".into(),
            requested: 10.0,
            available: 5.0,
        };
        assert!(e.to_string().contains("a1"));
        assert!(e.to_string().contains("10"));
    }

    #[test]
    fn test_reallocation_failed_display() {
        let e = BudgetError::ReallocationFailed {
            from: "a1".into(),
            to: "a2".into(),
            amount: 1.0,
            available: 0.5,
        };
        assert!(e.to_string().contains("a1"));
        assert!(e.to_string().contains("a2"));
    }

    #[test]
    fn test_unknown_agent_display() {
        let e = BudgetError::UnknownAgent("ghost-agent".into());
        assert!(e.to_string().contains("ghost-agent"));
    }

    #[test]
    fn test_persistence_error_display() {
        let e = BudgetError::PersistenceError("disk full".into());
        assert!(e.to_string().contains("disk full"));
    }

    #[test]
    fn test_invalid_config_display() {
        let e = BudgetError::InvalidConfig("limit must be positive".into());
        assert!(e.to_string().contains("positive"));
    }

    #[test]
    fn test_budget_error_is_clone() {
        let e = BudgetError::UnknownAgent("x".into());
        let _ = e.clone();
    }

    #[test]
    fn test_budget_error_is_partial_eq() {
        let a = BudgetError::UnknownAgent("x".into());
        let b = BudgetError::UnknownAgent("x".into());
        assert_eq!(a, b);
    }
}
