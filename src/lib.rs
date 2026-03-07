//! # llm-budget
//!
//! Autonomous cost governance primitives for LLM agent fleets.
//! Provides hard budget enforcement with atomic tracking, model downgrade
//! cascades, kill switches, and fleet-wide allocation — zero silent failures.

pub mod budget;
pub mod error;
pub mod fleet;
pub mod model;
pub mod persistence;

pub use budget::{AgentBudget, BudgetEnvelope, BudgetSnapshot, SpendRecord};
pub use error::BudgetError;
pub use fleet::{FleetAllocator, FleetBudget};
pub use model::{ModelCascade, ModelTier};
pub use persistence::BudgetStore;
