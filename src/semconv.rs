//! Names from the OpenTelemetry GenAI cost conventions.
//!
//! These follow `gen_ai.usage.cost.*` as proposed in
//! [open-telemetry/semantic-conventions-genai#443](https://github.com/open-telemetry/semantic-conventions-genai/pull/443)
//! (still open, stability "development", as of November 2026). The A2A budget
//! extension draft (a2aproject/A2A#2121) uses the same names and the same
//! `source` values, so one set of values fills both. If the convention
//! changes before it merges, these constants follow it in a minor release.
//!
//! The convention scopes a cost to one operation: it MUST NOT include cost
//! attributable to nested operations. llm-budget keeps a scope's own spend
//! ([`AgentBudget::own_cost_usd`](crate::AgentBudget::own_cost_usd)) apart
//! from its descendants' ([`AgentBudget::child_cost_usd`](crate::AgentBudget::child_cost_usd))
//! for that reason.

/// `gen_ai.usage.cost.amount`: the cost of one operation, in the currency of
/// [`COST_CURRENCY`].
pub const COST_AMOUNT: &str = "gen_ai.usage.cost.amount";

/// `gen_ai.usage.cost.currency`: an ISO 4217 code. llm-budget prices in
/// `USD`.
pub const COST_CURRENCY: &str = "gen_ai.usage.cost.currency";

/// `gen_ai.usage.cost.source`: [`CostSource::Provider`] or
/// [`CostSource::Local`].
pub const COST_SOURCE: &str = "gen_ai.usage.cost.source";

/// The attribute name llm-budget 0.5 and 0.6 emitted, before the
/// convention was proposed. Still read on spans, never written.
pub const LEGACY_COST: &str = "gen_ai.usage.cost";

/// The currency every amount in this crate is in.
pub const USD: &str = "USD";

/// Where a cost figure came from (`gen_ai.usage.cost.source`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[serde(rename_all = "lowercase")]
pub enum CostSource {
    /// The cost was in the response the client received (from the provider
    /// or a gateway that priced the call).
    Provider,
    /// The emitting component computed it from token counts and prices it
    /// owns (llm-budget's price table).
    Local,
}

impl CostSource {
    /// `provider` or `local`.
    pub fn as_str(self) -> &'static str {
        match self {
            CostSource::Provider => "provider",
            CostSource::Local => "local",
        }
    }

    /// Parse `provider` or `local` (exact, as the convention spells them).
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "provider" => Some(CostSource::Provider),
            "local" => Some(CostSource::Local),
            _ => None,
        }
    }
}

impl std::fmt::Display for CostSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}
