#![doc = include_str!("../README.md")]
//!
//! # Where to go next
//!
//! - **Calls you price after they happen** (most of them): authorize with
//!   [`AgentBudget::reserve`] before the call and book the real cost with
//!   [`Reservation::settle`] after it, or [`AgentBudget::charge_usd_force`]
//!   when nothing was reserved. Booked spend always matches the bill; see
//!   [`budget`] for the full picture.
//! - **Calls you price before making them**: [`AgentBudget::charge_model`],
//!   [`AgentBudget::charge_usage`] and friends check and book in one step.
//! - **Memory**: each agent keeps its last 1,024 [`SpendRecord`]s by default.
//!   Choose with [`AgentBudget::builder`] and [`AuditLog`], and ship records
//!   out with [`AgentBudget::drain_audit_log`].
//! - **Fleets**: [`FleetAllocator`] splits one total into per-agent caps.
//! - **Prices**: [`PriceTable`] (232 models) and [`Usage`] for provider
//!   usage blocks with prompt-cache counts.
//! - **rig agents** (feature `rig`): `rig::BudgetHook`, with a comparison to
//!   rig's own token-count budget example in its module docs.
#![cfg_attr(docsrs, feature(doc_cfg))]
#![warn(missing_docs)]

pub mod budget;
pub mod error;
pub mod fleet;
pub mod model;
pub mod observe;
pub mod persistence;
pub mod pricing;
#[cfg(feature = "ratelimit")]
#[cfg_attr(docsrs, doc(cfg(feature = "ratelimit")))]
pub mod ratelimit;
#[cfg(feature = "rig")]
#[cfg_attr(docsrs, doc(cfg(feature = "rig")))]
pub mod rig;
#[cfg(feature = "tokens")]
#[cfg_attr(docsrs, doc(cfg(feature = "tokens")))]
pub mod tokens;
pub mod usage;

pub use budget::{
    AgentBudget, AgentBudgetBuilder, AuditLog, BudgetEnvelope, BudgetSnapshot, ChargeContext, ChargeOutcome,
    ChargeReceipt, Reservation, SpendRecord,
};
pub use error::BudgetError;
pub use fleet::{FleetAllocator, FleetBudget};
pub use model::{ModelCascade, ModelTier};
pub use persistence::BudgetStore;
pub use pricing::{ModelPrice, PriceTable};
pub use usage::Usage;
