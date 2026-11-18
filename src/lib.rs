#![doc = include_str!("../README.md")]
//!
//! # Where to go next
//!
//! - **Any client, any framework** (the core pattern): before the call,
//!   [`AgentBudget::reserve_for`] holds the estimated cost (or refuses, so the
//!   call is never made); after it, [`Reservation::settle_usage`] books what
//!   the provider reported as a [`Usage`] (input, output, cache read and
//!   cache write tokens). [`AgentBudget::record`] books a finished call when
//!   nothing was reserved. Booked spend always matches the bill; see
//!   [`budget`] for the full picture, and [`AgentBudget::reserve`] /
//!   [`Reservation::settle`] for dollar amounts you price yourself.
//! - **Calls you price before making them**: [`AgentBudget::charge_model`],
//!   [`AgentBudget::charge_usage`] and friends check and book in one step.
//! - **Memory**: each agent keeps its last 1,024 [`SpendRecord`]s by default.
//!   Choose with [`AgentBudget::builder`] and [`AuditLog`], and ship records
//!   out with [`AgentBudget::drain_audit_log`].
//! - **Sub-agents and parallel fan-out**: [`AgentBudget::child`] reserves a
//!   slice of a budget for a sub-agent ([`Limit::Usd`] or
//!   [`Limit::Fraction`]); everything the child books is booked on its
//!   ancestors at once, and closing its [`ScopeGuard`] refunds the rest, so
//!   parallel children can never be admitted for more than the parent has
//!   left. See [`budget::scope`].
//! - **Across processes**: [`context::BudgetContext`] carries a slice down
//!   to a remote agent or tool (A2A metadata, W3C baggage, MCP `_meta`) and
//!   its cost back up, in the shape of the A2A budget extension draft.
//! - **Models with no price**: [`UnpricedPolicy`] picks a [`PriceAction`]
//!   for each [`PriceStatus`] (`Unpriced`, `Missing`, `NoCatalog`).
//! - **Fleets**: [`FleetAllocator`] splits one total into per-agent caps.
//! - **Prices**: [`PriceTable`]: the [llm-prices](https://docs.rs/llm-prices)
//!   crate (dated, source-cited, long-context tiers) over a LiteLLM snapshot
//!   for other providers, with your own prices or a hook on top; and [`Usage`]
//!   for provider usage blocks with prompt-cache counts.
//! - **rig agents** (feature `rig`): `rig::BudgetHook`, with a comparison to
//!   rig's own token-count budget example in its module docs.
//! - **genai** (feature `genai`): `genai::exec_chat` and
//!   `genai::exec_chat_stream` run a genai `Client` call under a hold.
//! - **metrics** (feature `metrics-layer`): `metrics_layer::CostLayer` wraps
//!   any metrics recorder and prices token counters a framework already emits
//!   (swiftide's preset, or any counter labelled with the model) into a USD
//!   metric, and can book them on an [`AgentBudget`].
//! - **async-openai** (feature `async-openai`): `openai::settle_response` and
//!   `openai::StreamUsage` settle from a chat completion or a stream's final
//!   usage chunk.
//! - **Traced apps** (feature `tracing-layer`): `tracing_layer::CostLayer`
//!   prices every span that carries OpenTelemetry GenAI usage fields (rig
//!   records them on each model call), emits `gen_ai.usage.cost.amount` /
//!   `.currency` / `.source` (the proposed OpenTelemetry GenAI cost names) and can
//!   book the spend on a capped budget, with no change to the traced code.
//! - **No async runtime required**: everything except `BudgetStore`
//!   (feature `store`, on by default) is runtime-free; the `async` charge
//!   methods never await anything runtime-specific. Turn default features off
//!   to drop tokio.
#![cfg_attr(docsrs, feature(doc_cfg))]
#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod budget;
pub mod context;
pub mod error;
pub mod fleet;
#[cfg(feature = "metrics-layer")]
#[cfg_attr(docsrs, doc(cfg(feature = "metrics-layer")))]
pub mod metrics_layer;
pub mod model;
pub mod observe;
#[cfg(feature = "async-openai")]
#[cfg_attr(docsrs, doc(cfg(feature = "async-openai")))]
pub mod openai;
#[cfg(feature = "genai")]
#[cfg_attr(docsrs, doc(cfg(feature = "genai")))]
pub mod genai;
#[cfg(feature = "store")]
#[cfg_attr(docsrs, doc(cfg(feature = "store")))]
pub mod persistence;
pub mod pricing;
#[cfg(feature = "ratelimit")]
#[cfg_attr(docsrs, doc(cfg(feature = "ratelimit")))]
pub mod ratelimit;
pub mod semconv;
#[cfg(feature = "rig")]
#[cfg_attr(docsrs, doc(cfg(feature = "rig")))]
pub mod rig;
#[cfg(feature = "tokens")]
#[cfg_attr(docsrs, doc(cfg(feature = "tokens")))]
pub mod tokens;
#[cfg(feature = "tracing-layer")]
#[cfg_attr(docsrs, doc(cfg(feature = "tracing-layer")))]
pub mod tracing_layer;
#[cfg(feature = "tower")]
#[cfg_attr(docsrs, doc(cfg(feature = "tower")))]
pub mod tower_layer;
pub mod usage;

pub use budget::{
    AgentBudget, AgentBudgetBuilder, AuditLog, BudgetEnvelope, BudgetScope, BudgetSnapshot, ChargeContext,
    ChargeOutcome, ChargeReceipt, Limit, Reservation, ScopeGuard, ScopeReceipt, SpendRecord,
};
pub use error::BudgetError;
pub use fleet::{FleetAllocator, FleetBudget};
pub use model::{ModelCascade, ModelTier};
#[cfg(feature = "store")]
#[cfg_attr(docsrs, doc(cfg(feature = "store")))]
pub use persistence::BudgetStore;
pub use pricing::{ModelPrice, PriceAction, PriceStatus, PriceTable, UnpricedPolicy};
pub use usage::Usage;
