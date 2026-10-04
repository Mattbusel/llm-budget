#![doc = include_str!("../README.md")]
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
#[cfg(feature = "tokens")]
#[cfg_attr(docsrs, doc(cfg(feature = "tokens")))]
pub mod tokens;
pub mod usage;

pub use budget::{AgentBudget, BudgetEnvelope, BudgetSnapshot, ChargeReceipt, SpendRecord};
pub use error::BudgetError;
pub use fleet::{FleetAllocator, FleetBudget};
pub use model::{ModelCascade, ModelTier};
pub use persistence::BudgetStore;
pub use pricing::{ModelPrice, PriceTable};
pub use usage::Usage;
