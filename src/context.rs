//! Carry a budget across a delegation boundary: what a caller allows a
//! callee to spend, and what the callee reports it spent.
//!
//! [`BudgetContext`] has the shape of the A2A budget extension draft
//! ([a2aproject/A2A#2121](https://github.com/a2aproject/A2A/issues/2121)):
//! `budget{cost, tokens, deadline}` going down, `incurred{cost,
//! aggregatedChildCost, tokens, delegations}` coming back, plus an optional
//! `reservation{id, amount, currency, expires}` for the slice the caller
//! reserved with [`AgentBudget::child`]. Three carriers move it:
//!
//! - **A2A** ([`a2a`]): the whole context under [`a2a::EXTENSION_URI`] in
//!   message, task or request metadata.
//! - **W3C baggage** ([`baggage`]): the outbound part as `llm.budget.*`
//!   members of a `baggage` header, so anything that already propagates
//!   OpenTelemetry baggage carries it.
//! - **MCP** ([`mcp`]): MCP reserves the `baggage` key of `params._meta` for
//!   W3C baggage (SEP-414), so the same members ride on any MCP request.
//!
//! Rules from the draft's discussion, applied here:
//!
//! - **Absent is not zero.** A figure nobody measured is omitted (`None`),
//!   never written as 0.
//! - **Money is a decimal string** on the wire (`"0.25"`), exact to the
//!   nano-unit; JSON numbers are accepted on input.
//! - `incurred.cost` is the reporter's **own** spend; `aggregatedChildCost`
//!   is its delegates'. In a stream, `incurred` is **cumulative**:
//!   [`IncurredBooker`] books only the growth.
//!
//! The Python package [`agent-budget`](https://pypi.org/project/agent-budget/)
//! produces the same JSON and baggage, key for key; both test suites read
//! the golden fixtures in `tests/fixtures/budget_context/`.
//!
//! ```
//! use llm_budget::context::{a2a, baggage, BudgetContext};
//! use llm_budget::{AgentBudget, Limit};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Caller: reserve a slice for the remote agent and describe it.
//! let parent = AgentBudget::builder("orchestrator", 1.00).build()?;
//! let slice = parent.child("remote-researcher", Limit::Usd(0.25))?;
//! let ctx = BudgetContext::outbound(&slice);
//! let metadata = a2a::to_metadata(&ctx)?;  // into SendMessageRequest.metadata
//! let header = baggage::encode(&ctx);      // or into a `baggage` header
//! assert!(header.starts_with("llm.budget.cost.amount=0.25,llm.budget.cost.currency=USD"));
//!
//! // Callee: a budget of exactly what it was given.
//! let received = a2a::from_metadata(&metadata)?.expect("budget extension present");
//! let callee = AgentBudget::from_context("researcher", &received)?;
//! assert!((callee.limit_usd() - 0.25).abs() < 1e-12);
//! callee.reserve(0.05)?.settle(0.04);
//!
//! // Callee reports back; caller books what it reported on the slice.
//! let report = BudgetContext::reporting(&callee);
//! let mut booker = llm_budget::context::IncurredBooker::new();
//! booker.book(&slice, report.incurred.as_ref());
//! assert!((parent.spent_usd() - 0.04).abs() < 1e-12);
//! # Ok(())
//! # }
//! ```

use std::fmt;

use serde::de::{self, Deserializer};
use serde::ser::Serializer;
use serde::{Deserialize, Serialize};

use crate::budget::{nanos_to_usd, AgentBudget, ChargeContext, ChargeOutcome, ScopeGuard};
use crate::error::BudgetError;
use crate::semconv::CostSource;

const NANOS: u64 = 1_000_000_000;

/// An exact, non-negative amount in whole nano-units (1 unit = 10^9 nanos),
/// written on the wire as a canonical decimal string: no sign, no exponent,
/// at most nine fractional digits, no trailing zeros (`"0.25"`, `"3"`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct Amount(pub u64);

impl Amount {
    /// From USD as a float, rounded half up to the nano-unit. `None` for
    /// negative, NaN, infinite or too large values.
    pub fn from_usd(usd: f64) -> Option<Self> {
        if !usd.is_finite() || usd < 0.0 {
            return None;
        }
        let n = (usd * NANOS as f64 + 0.5).floor();
        if n >= u64::MAX as f64 {
            return None;
        }
        Some(Amount(n as u64))
    }

    /// The amount as a float (for arithmetic and display, not the wire).
    pub fn as_f64(self) -> f64 {
        nanos_to_usd(self.0)
    }

    /// Parse the wire form. Accepts the canonical decimal string and any
    /// decimal with up to nine fractional digits.
    pub fn parse(s: &str) -> Result<Self, ContextError> {
        let bad = || ContextError(format!("not a decimal amount: {s:?}"));
        let (whole, frac) = match s.split_once('.') {
            Some((w, f)) => (w, Some(f)),
            None => (s, None),
        };
        let digits = |x: &str| !x.is_empty() && x.bytes().all(|b| b.is_ascii_digit());
        if !digits(whole) || whole.len() > 20 || frac.is_some_and(|f| !digits(f) || f.len() > 9) {
            return Err(bad());
        }
        let w: u64 = whole.parse().map_err(|_| bad())?;
        let f: u64 = match frac {
            Some(f) => format!("{f:0<9}").parse().map_err(|_| bad())?,
            None => 0,
        };
        w.checked_mul(NANOS).and_then(|x| x.checked_add(f)).map(Amount).ok_or_else(bad)
    }
}

impl fmt::Display for Amount {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (whole, frac) = (self.0 / NANOS, self.0 % NANOS);
        if frac == 0 {
            write!(f, "{whole}")
        } else {
            let s = format!("{frac:09}");
            write!(f, "{whole}.{}", s.trim_end_matches('0'))
        }
    }
}

impl Serialize for Amount {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.collect_str(self)
    }
}

impl<'de> Deserialize<'de> for Amount {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        struct V;
        impl de::Visitor<'_> for V {
            type Value = Amount;
            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("a decimal string or a non-negative number")
            }
            fn visit_str<E: de::Error>(self, v: &str) -> Result<Amount, E> {
                Amount::parse(v).map_err(E::custom)
            }
            fn visit_u64<E: de::Error>(self, v: u64) -> Result<Amount, E> {
                v.checked_mul(NANOS).map(Amount).ok_or_else(|| E::custom("amount too large"))
            }
            fn visit_i64<E: de::Error>(self, v: i64) -> Result<Amount, E> {
                u64::try_from(v).map_err(|_| E::custom("amount must be >= 0")).and_then(|u| self.visit_u64(u))
            }
            fn visit_f64<E: de::Error>(self, v: f64) -> Result<Amount, E> {
                Amount::from_usd(v).ok_or_else(|| E::custom("amount must be a finite number >= 0"))
            }
        }
        d.deserialize_any(V)
    }
}

/// A whole count that also reads whole floats (a protobuf `Struct` stores
/// every number as a double).
fn de_count<'de, D: Deserializer<'de>>(d: D) -> Result<Option<u64>, D::Error> {
    struct V;
    impl<'de> de::Visitor<'de> for V {
        type Value = Option<u64>;
        fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.write_str("a whole number >= 0")
        }
        fn visit_none<E: de::Error>(self) -> Result<Self::Value, E> {
            Ok(None)
        }
        fn visit_unit<E: de::Error>(self) -> Result<Self::Value, E> {
            Ok(None)
        }
        fn visit_some<D2: Deserializer<'de>>(self, d: D2) -> Result<Self::Value, D2::Error> {
            d.deserialize_any(self)
        }
        fn visit_u64<E: de::Error>(self, v: u64) -> Result<Self::Value, E> {
            Ok(Some(v))
        }
        fn visit_i64<E: de::Error>(self, v: i64) -> Result<Self::Value, E> {
            u64::try_from(v).map(Some).map_err(|_| E::custom("count must be >= 0"))
        }
        fn visit_f64<E: de::Error>(self, v: f64) -> Result<Self::Value, E> {
            if v.is_finite() && v >= 0.0 && v.fract() == 0.0 && v < u64::MAX as f64 {
                Ok(Some(v as u64))
            } else {
                Err(E::custom("count must be a whole number >= 0"))
            }
        }
    }
    d.deserialize_option(V)
}

fn de_currency<'de, D: Deserializer<'de>>(d: D) -> Result<String, D::Error> {
    let s = String::deserialize(d)?;
    if s.len() == 3 && s.bytes().all(|b| b.is_ascii_uppercase()) {
        Ok(s)
    } else {
        Err(de::Error::custom(format!("not an ISO 4217 currency code: {s:?}")))
    }
}

fn de_bounded<'de, D: Deserializer<'de>, const MAX: usize>(d: D) -> Result<Option<String>, D::Error> {
    let s = Option::<String>::deserialize(d)?;
    match s {
        Some(v) if v.is_empty() || v.len() > MAX => {
            Err(de::Error::custom(format!("expected a non-empty string of at most {MAX} bytes")))
        }
        other => Ok(other),
    }
}

fn usd() -> String {
    crate::semconv::USD.to_string()
}

/// An amount and its currency.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct Money {
    /// The amount.
    pub amount: Amount,
    /// ISO 4217 code.
    #[serde(deserialize_with = "de_currency")]
    pub currency: String,
}

impl Money {
    /// An amount in US dollars.
    pub fn usd(amount: Amount) -> Self {
        Self { amount, currency: usd() }
    }
}

/// A reported cost: amount, currency and, when the reporter says,
/// `provider` (from a response) or `local` (computed from prices).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct Cost {
    /// The amount.
    pub amount: Amount,
    /// ISO 4217 code.
    #[serde(deserialize_with = "de_currency")]
    pub currency: String,
    /// Where the figure came from; `None` when not reported.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<CostSource>,
}

impl Cost {
    /// A cost in US dollars.
    pub fn usd(amount: Amount, source: Option<CostSource>) -> Self {
        Self { amount, currency: usd(), source }
    }
}

/// Token limit going down.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[non_exhaustive]
pub struct TokenLimit {
    /// The most tokens the callee may use.
    #[serde(default, deserialize_with = "de_count", skip_serializing_if = "Option::is_none")]
    pub limit: Option<u64>,
}

/// Token counts coming back. `total` is what the provider reported and is
/// never derived from the others.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
#[non_exhaustive]
pub struct TokenCounts {
    /// Prompt tokens.
    #[serde(default, deserialize_with = "de_count", skip_serializing_if = "Option::is_none")]
    pub input: Option<u64>,
    /// Completion tokens.
    #[serde(default, deserialize_with = "de_count", skip_serializing_if = "Option::is_none")]
    pub output: Option<u64>,
    /// Prompt tokens read from a cache.
    #[serde(default, alias = "cache_read", deserialize_with = "de_count", skip_serializing_if = "Option::is_none")]
    pub cache_read: Option<u64>,
    /// Prompt tokens written to a cache.
    #[serde(default, alias = "cache_write", deserialize_with = "de_count", skip_serializing_if = "Option::is_none")]
    pub cache_write: Option<u64>,
    /// The provider's own total.
    #[serde(default, deserialize_with = "de_count", skip_serializing_if = "Option::is_none")]
    pub total: Option<u64>,
}

/// What the caller allows.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
#[non_exhaustive]
pub struct BudgetLimit {
    /// Money the callee may spend.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cost: Option<Money>,
    /// Tokens the callee may use.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tokens: Option<TokenLimit>,
    /// RFC 3339 time to finish by (opaque here).
    #[serde(default, deserialize_with = "de_bounded::<_, 64>", skip_serializing_if = "Option::is_none")]
    pub deadline: Option<String>,
}

/// What the callee spent.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
#[non_exhaustive]
pub struct Incurred {
    /// The callee's own spend, without its delegates'.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cost: Option<Cost>,
    /// Everything the callee's delegates reported (their own cost plus
    /// their own aggregate). The draft calls this name provisional.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub aggregated_child_cost: Option<Cost>,
    /// Token counts.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tokens: Option<TokenCounts>,
    /// How many delegations the callee made.
    #[serde(default, deserialize_with = "de_count", skip_serializing_if = "Option::is_none")]
    pub delegations: Option<u64>,
}

impl Incurred {
    /// `cost` plus `aggregated_child_cost` in `currency`, or `None` when
    /// neither is reported in it (absent is not zero).
    pub fn total(&self, currency: &str) -> Option<Amount> {
        let parts: Vec<u64> = [&self.cost, &self.aggregated_child_cost]
            .into_iter()
            .flatten()
            .filter(|c| c.currency == currency)
            .map(|c| c.amount.0)
            .collect();
        (!parts.is_empty()).then(|| Amount(parts.iter().fold(0u64, |a, b| a.saturating_add(*b))))
    }
}

/// The slice a caller reserved for this delegation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ReservationInfo {
    /// Opaque id, unique per reservation (never shared by siblings).
    #[serde(deserialize_with = "de_id")]
    pub id: String,
    /// The amount reserved.
    pub amount: Amount,
    /// ISO 4217 code.
    #[serde(deserialize_with = "de_currency")]
    pub currency: String,
    /// RFC 3339 time the reservation lapses (opaque here).
    #[serde(default, deserialize_with = "de_bounded::<_, 64>", skip_serializing_if = "Option::is_none")]
    pub expires: Option<String>,
}

fn de_id<'de, D: Deserializer<'de>>(d: D) -> Result<String, D::Error> {
    let s = String::deserialize(d)?;
    if s.is_empty() || s.len() > 128 {
        return Err(de::Error::custom("reservation id must be 1 to 128 bytes"));
    }
    Ok(s)
}

/// The extension payload; see the [module docs](self).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
#[non_exhaustive]
pub struct BudgetContext {
    /// What the caller allows.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub budget: Option<BudgetLimit>,
    /// What the callee spent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub incurred: Option<Incurred>,
    /// The slice the caller reserved.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reservation: Option<ReservationInfo>,
}

/// A budget context that could not be read or built.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContextError(pub String);

impl fmt::Display for ContextError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "budget context: {}", self.0)
    }
}

impl std::error::Error for ContextError {}

impl From<ContextError> for BudgetError {
    fn from(e: ContextError) -> Self {
        BudgetError::InvalidConfig(e.to_string())
    }
}

/// Every object in a context must be a JSON object (or null). serde would
/// otherwise read a struct from an array of its fields.
fn check_shape(value: &serde_json::Value) -> Result<(), ContextError> {
    use serde_json::Value;
    fn obj<'a>(v: Option<&'a Value>, path: &str) -> Result<Option<&'a serde_json::Map<String, Value>>, ContextError> {
        match v {
            None | Some(Value::Null) => Ok(None),
            Some(Value::Object(m)) => Ok(Some(m)),
            Some(_) => Err(ContextError(format!("{path} must be an object"))),
        }
    }
    let Some(top) = obj(Some(value), "budget context")? else {
        return Err(ContextError("budget context must be an object".into()));
    };
    if let Some(b) = obj(top.get("budget"), "budget")? {
        obj(b.get("cost"), "budget.cost")?;
        obj(b.get("tokens"), "budget.tokens")?;
    }
    if let Some(i) = obj(top.get("incurred"), "incurred")? {
        obj(i.get("cost"), "incurred.cost")?;
        obj(i.get("aggregatedChildCost"), "incurred.aggregatedChildCost")?;
        obj(i.get("tokens"), "incurred.tokens")?;
    }
    obj(top.get("reservation"), "reservation")?;
    Ok(())
}

/// A random-looking, unique reservation id (`rsv_` + 32 hex digits).
fn new_reservation_id() -> String {
    use std::collections::hash_map::RandomState;
    use std::hash::{BuildHasher, Hasher};
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let t = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let mut out = String::from("rsv_");
    for salt in 0..2u64 {
        let mut h = RandomState::new().build_hasher();
        h.write_u64(n);
        h.write_u128(t);
        h.write_u64(salt);
        out.push_str(&format!("{:016x}", h.finish()));
    }
    out
}

impl BudgetContext {
    /// Read a context from a JSON value. Unknown keys are ignored.
    ///
    /// # Errors
    /// [`ContextError`] for a malformed known field.
    pub fn from_json(value: &serde_json::Value) -> Result<Self, ContextError> {
        check_shape(value)?;
        serde_json::from_value(value.clone()).map_err(|e| ContextError(e.to_string()))
    }

    /// The context as a JSON value.
    ///
    /// # Errors
    /// Never in practice; serde_json reports it if it happens.
    pub fn to_json(&self) -> Result<serde_json::Value, ContextError> {
        serde_json::to_value(self).map_err(|e| ContextError(e.to_string()))
    }

    /// The context to send with a delegation made from `scope`:
    /// `budget.cost` is what the scope has left, and `reservation` is its
    /// hold in the parent with a fresh id.
    pub fn outbound(scope: &ScopeGuard) -> Self {
        let remaining = Amount(scope.remaining_nanos());
        let held = Amount(scope.held_nanos());
        Self {
            budget: Some(BudgetLimit { cost: Some(Money::usd(remaining)), tokens: None, deadline: None }),
            incurred: None,
            reservation: Some(ReservationInfo { id: new_reservation_id(), amount: held, currency: usd(), expires: None }),
        }
    }

    /// A budget-only context (no reservation): `budget.cost` is what
    /// `budget` has left.
    pub fn remaining(budget: &AgentBudget) -> Self {
        let remaining = Amount(budget.remaining_nanos());
        Self {
            budget: Some(BudgetLimit { cost: Some(Money::usd(remaining)), tokens: None, deadline: None }),
            ..Self::default()
        }
    }

    /// The report a callee sends back: [`Incurred::of`] `budget`.
    pub fn reporting(budget: &AgentBudget) -> Self {
        Self { incurred: Some(Incurred::of(budget, Some(CostSource::Local))), ..Self::default() }
    }

    /// Set `budget.deadline` (RFC 3339).
    pub fn with_deadline(mut self, deadline: impl Into<String>) -> Self {
        self.budget.get_or_insert_with(BudgetLimit::default).deadline = Some(deadline.into());
        self
    }

    /// Set `budget.tokens.limit`.
    pub fn with_token_limit(mut self, limit: u64) -> Self {
        self.budget.get_or_insert_with(BudgetLimit::default).tokens = Some(TokenLimit { limit: Some(limit) });
        self
    }

    /// The money this context allows in USD: `budget.cost`, else the
    /// reservation amount.
    pub fn budget_usd(&self) -> Option<Amount> {
        let from_budget = self.budget.as_ref().and_then(|b| b.cost.as_ref()).filter(|m| m.currency == crate::semconv::USD);
        if let Some(m) = from_budget {
            return Some(m.amount);
        }
        self.reservation.as_ref().filter(|r| r.currency == crate::semconv::USD).map(|r| r.amount)
    }
}

impl Incurred {
    /// What `budget` reports: its own spend as `cost` (with `source`), its
    /// descendants' as `aggregatedChildCost` (only when it delegated), and
    /// how many child scopes it created.
    pub fn of(budget: &AgentBudget, source: Option<CostSource>) -> Self {
        let delegations = budget.delegations();
        let own = Amount(budget.own_cost_nanos());
        let child = Amount(budget.child_cost_nanos());
        Self {
            cost: Some(Cost::usd(own, source)),
            aggregated_child_cost: (delegations > 0).then(|| Cost::usd(child, None)),
            tokens: None,
            delegations: Some(delegations),
        }
    }
}

impl AgentBudget {
    /// A root budget for a callee, capped at what `ctx` allows in USD
    /// (`budget.cost`, else the reservation amount).
    ///
    /// # Errors
    /// [`BudgetError::InvalidConfig`] when `ctx` has no USD amount or it is
    /// zero.
    pub fn from_context(agent_id: impl Into<String>, ctx: &BudgetContext) -> Result<Self, BudgetError> {
        let amount = ctx
            .budget_usd()
            .ok_or_else(|| BudgetError::InvalidConfig("budget context has no USD amount".into()))?;
        AgentBudget::builder(agent_id, amount.as_f64()).build()
    }
}

/// Books a remote callee's reported cost on a local budget exactly once,
/// even when the report arrives many times (stream updates, a resent final
/// task): `incurred` is cumulative, so only its growth is booked.
#[derive(Debug, Default, Clone)]
pub struct IncurredBooker {
    booked: u64,
}

impl IncurredBooker {
    /// A booker that has booked nothing yet.
    pub fn new() -> Self {
        Self::default()
    }

    /// What has been booked so far.
    pub fn booked(&self) -> Amount {
        Amount(self.booked)
    }

    /// Book the part of `incurred`'s USD total not booked yet, as spend that
    /// already happened (it may cross the cap). `None` when nothing new was
    /// reported.
    pub fn book(&mut self, budget: &AgentBudget, incurred: Option<&Incurred>) -> Option<ChargeOutcome> {
        let total = incurred?.total(crate::semconv::USD)?.0;
        if total <= self.booked {
            return None;
        }
        let delta = total - self.booked;
        self.booked = total;
        Some(budget.charge_usd_force(nanos_to_usd(delta), ChargeContext::new()))
    }
}

/// W3C baggage: the outbound part of a context as `llm.budget.*` members.
pub mod baggage {
    use std::collections::BTreeMap;

    use super::{Amount, BudgetContext, BudgetLimit, ContextError, Money, ReservationInfo, TokenLimit};

    /// Prefix of every member this module reads or writes.
    pub const PREFIX: &str = "llm.budget.";
    /// Longest header [`decode`] accepts (the W3C baggage limit).
    pub const MAX_BYTES: usize = 8192;
    /// `budget.cost.amount`.
    pub const COST_AMOUNT: &str = "llm.budget.cost.amount";
    /// `budget.cost.currency`.
    pub const COST_CURRENCY: &str = "llm.budget.cost.currency";
    /// `budget.tokens.limit`.
    pub const TOKENS_LIMIT: &str = "llm.budget.tokens.limit";
    /// `budget.deadline`.
    pub const DEADLINE: &str = "llm.budget.deadline";
    /// `reservation.id`.
    pub const RESERVATION_ID: &str = "llm.budget.reservation.id";
    /// `reservation.amount`.
    pub const RESERVATION_AMOUNT: &str = "llm.budget.reservation.amount";
    /// `reservation.currency`.
    pub const RESERVATION_CURRENCY: &str = "llm.budget.reservation.currency";
    /// `reservation.expires`.
    pub const RESERVATION_EXPIRES: &str = "llm.budget.reservation.expires";

    fn safe(b: u8) -> bool {
        // W3C baggage-octet, minus '%' which starts an escape.
        matches!(b, 0x21 | 0x23..=0x2B | 0x2D..=0x3A | 0x3C..=0x5B | 0x5D..=0x7E) && b != b'%'
    }

    fn token(b: u8) -> bool {
        b.is_ascii_alphanumeric() || b"!#$%&'*+-.^_`|~".contains(&b)
    }

    fn encode_value(v: &str) -> String {
        let mut out = String::with_capacity(v.len());
        for b in v.bytes() {
            if safe(b) {
                out.push(char::from(b));
            } else {
                out.push_str(&format!("%{b:02X}"));
            }
        }
        out
    }

    fn decode_value(v: &str, key: &str) -> Result<String, ContextError> {
        let bytes = v.as_bytes();
        let mut out = Vec::with_capacity(bytes.len());
        let mut i = 0;
        while i < bytes.len() {
            let b = bytes[i];
            if b == b'%' {
                let hex = bytes.get(i + 1..i + 3).filter(|h| h.iter().all(u8::is_ascii_hexdigit));
                let hex = hex.ok_or_else(|| ContextError(format!("bad percent escape in baggage value of {key}")))?;
                let s = std::str::from_utf8(hex).map_err(|e| ContextError(e.to_string()))?;
                out.push(u8::from_str_radix(s, 16).map_err(|e| ContextError(e.to_string()))?);
                i += 3;
            } else if safe(b) {
                out.push(b);
                i += 1;
            } else {
                return Err(ContextError(format!("byte 0x{b:02x} not allowed in baggage value of {key}")));
            }
        }
        String::from_utf8(out).map_err(|_| ContextError(format!("baggage value of {key} is not UTF-8")))
    }

    /// The `llm.budget.*` members of `ctx`, in a fixed order (budget, then
    /// reservation; `incurred` travels back in A2A metadata, not baggage).
    pub fn members(ctx: &BudgetContext) -> Vec<(&'static str, String)> {
        let mut out = Vec::new();
        if let Some(b) = &ctx.budget {
            if let Some(c) = &b.cost {
                out.push((COST_AMOUNT, c.amount.to_string()));
                out.push((COST_CURRENCY, c.currency.clone()));
            }
            if let Some(limit) = b.tokens.and_then(|t| t.limit) {
                out.push((TOKENS_LIMIT, limit.to_string()));
            }
            if let Some(d) = &b.deadline {
                out.push((DEADLINE, d.clone()));
            }
        }
        if let Some(r) = &ctx.reservation {
            out.push((RESERVATION_ID, r.id.clone()));
            out.push((RESERVATION_AMOUNT, r.amount.to_string()));
            out.push((RESERVATION_CURRENCY, r.currency.clone()));
            if let Some(e) = &r.expires {
                out.push((RESERVATION_EXPIRES, e.clone()));
            }
        }
        out
    }

    /// A `baggage` header value with only the budget members.
    pub fn encode(ctx: &BudgetContext) -> String {
        members(ctx).iter().map(|(k, v)| format!("{k}={}", encode_value(v))).collect::<Vec<_>>().join(",")
    }

    /// `existing` with every `llm.budget.*` member replaced by `ctx`'s;
    /// other members are kept as they were.
    pub fn merge(existing: Option<&str>, ctx: &BudgetContext) -> String {
        let mut parts: Vec<String> = existing
            .unwrap_or("")
            .split(',')
            .map(str::trim)
            .filter(|m| !m.is_empty())
            .filter(|m| !m.split('=').next().unwrap_or("").trim().starts_with(PREFIX))
            .map(str::to_string)
            .collect();
        let ours = encode(ctx);
        if !ours.is_empty() {
            parts.push(ours);
        }
        parts.join(",")
    }

    fn parse_members(header: &str) -> Result<BTreeMap<String, String>, ContextError> {
        if header.len() > MAX_BYTES {
            return Err(ContextError(format!("baggage header longer than {MAX_BYTES} bytes")));
        }
        let mut found = BTreeMap::new();
        for member in header.split(',') {
            let head = member.split(';').next().unwrap_or("");
            let Some((key, value)) = head.split_once('=') else { continue };
            let key = key.trim_matches([' ', '\t']);
            if !key.starts_with(PREFIX) {
                continue;
            }
            if !key.bytes().all(token) {
                return Err(ContextError(format!("baggage key {key:?} is not a token")));
            }
            let value = decode_value(value.trim_matches([' ', '\t']), key)?;
            if found.insert(key.to_string(), value).is_some() {
                return Err(ContextError(format!("baggage key {key} appears twice")));
            }
        }
        Ok(found)
    }

    fn currency(v: &str) -> Result<String, ContextError> {
        if v.len() == 3 && v.bytes().all(|b| b.is_ascii_uppercase()) {
            Ok(v.to_string())
        } else {
            Err(ContextError(format!("not an ISO 4217 currency code: {v:?}")))
        }
    }

    fn bounded(v: Option<&String>, key: &str, max: usize) -> Result<Option<String>, ContextError> {
        match v {
            Some(s) if s.len() > max => Err(ContextError(format!("{key} is longer than {max} characters"))),
            other => Ok(other.cloned()),
        }
    }

    /// Read the budget members of a `baggage` header: `Ok(None)` when it has
    /// none.
    ///
    /// # Errors
    /// [`ContextError`] when they are malformed (bad escapes, duplicate
    /// keys, a missing currency, a header over [`MAX_BYTES`]). Never panics.
    pub fn decode(header: &str) -> Result<Option<BudgetContext>, ContextError> {
        let m = parse_members(header)?;
        if m.is_empty() {
            return Ok(None);
        }
        let get = |k: &str| m.get(k);
        let cost = match (get(COST_AMOUNT), get(COST_CURRENCY)) {
            (Some(a), Some(c)) => Some(Money { amount: Amount::parse(a)?, currency: currency(c)? }),
            (None, None) => None,
            _ => return Err(ContextError("llm.budget.cost.amount and llm.budget.cost.currency go together".into())),
        };
        let limit = match get(TOKENS_LIMIT) {
            Some(v) if !v.is_empty() && v.len() <= 20 && v.bytes().all(|b| b.is_ascii_digit()) => {
                Some(v.parse::<u64>().map_err(|e| ContextError(format!("{TOKENS_LIMIT}: {e}")))?)
            }
            Some(v) => return Err(ContextError(format!("not a whole number: {v:?}"))),
            None => None,
        };
        let deadline = bounded(get(DEADLINE), DEADLINE, 64)?.filter(|d| !d.is_empty());
        let budget = (cost.is_some() || limit.is_some() || deadline.is_some()).then(|| BudgetLimit {
            cost,
            tokens: limit.map(|l| TokenLimit { limit: Some(l) }),
            deadline,
        });
        let any_res = [RESERVATION_ID, RESERVATION_AMOUNT, RESERVATION_CURRENCY, RESERVATION_EXPIRES]
            .iter()
            .any(|k| m.contains_key(*k));
        let reservation = if any_res {
            let need = |k: &str| get(k).ok_or_else(|| ContextError(format!("{k} is missing from the reservation")));
            let id = bounded(Some(need(RESERVATION_ID)?), RESERVATION_ID, 128)?.unwrap_or_default();
            if id.is_empty() {
                return Err(ContextError("llm.budget.reservation.id is empty".into()));
            }
            Some(ReservationInfo {
                id,
                amount: Amount::parse(need(RESERVATION_AMOUNT)?)?,
                currency: currency(need(RESERVATION_CURRENCY)?)?,
                expires: bounded(get(RESERVATION_EXPIRES), RESERVATION_EXPIRES, 64)?.filter(|e| !e.is_empty()),
            })
        } else {
            None
        };
        if budget.is_none() && reservation.is_none() {
            return Ok(None);
        }
        Ok(Some(BudgetContext { budget, incurred: None, reservation }))
    }
}

/// A2A: the whole context under the extension URI in metadata.
pub mod a2a {
    use super::{BudgetContext, ContextError};

    /// The extension URI proposed in a2aproject/A2A#2121. The A2A project has
    /// not reserved it yet; if it publishes a different one, this constant
    /// follows in a minor release and [`from_metadata`] keeps reading both.
    pub const EXTENSION_URI: &str = "https://a2a-protocol.org/extensions/budget/v1";

    /// `{EXTENSION_URI: ctx}`, ready to merge into message, task or request
    /// metadata.
    ///
    /// # Errors
    /// Only if serialization fails.
    pub fn to_metadata(ctx: &BudgetContext) -> Result<serde_json::Value, ContextError> {
        let mut map = serde_json::Map::new();
        map.insert(EXTENSION_URI.to_string(), ctx.to_json()?);
        Ok(serde_json::Value::Object(map))
    }

    /// Put `ctx` into an existing metadata object (replacing an earlier one).
    ///
    /// # Errors
    /// Only if serialization fails.
    pub fn insert(metadata: &mut serde_json::Map<String, serde_json::Value>, ctx: &BudgetContext) -> Result<(), ContextError> {
        metadata.insert(EXTENSION_URI.to_string(), ctx.to_json()?);
        Ok(())
    }

    /// The context under the extension URI: `Ok(None)` if there is none.
    ///
    /// # Errors
    /// [`ContextError`] when the payload is malformed.
    pub fn from_metadata(metadata: &serde_json::Value) -> Result<Option<BudgetContext>, ContextError> {
        match metadata.get(EXTENSION_URI) {
            None | Some(serde_json::Value::Null) => Ok(None),
            Some(v) => BudgetContext::from_json(v).map(Some),
        }
    }
}

/// MCP: the baggage members in `params._meta.baggage` (SEP-414).
pub mod mcp {
    use super::{baggage, BudgetContext, ContextError};

    /// Put `ctx` into an MCP `_meta` object's `baggage`, merged with baggage
    /// already there.
    pub fn inject(meta: &mut serde_json::Map<String, serde_json::Value>, ctx: &BudgetContext) {
        let existing = meta.get("baggage").and_then(serde_json::Value::as_str);
        let merged = baggage::merge(existing, ctx);
        if !merged.is_empty() {
            meta.insert("baggage".to_string(), serde_json::Value::String(merged));
        }
    }

    /// The budget members of an MCP `_meta` object's `baggage`.
    ///
    /// # Errors
    /// [`ContextError`] when they are malformed.
    pub fn extract(meta: &serde_json::Value) -> Result<Option<BudgetContext>, ContextError> {
        match meta.get("baggage") {
            None | Some(serde_json::Value::Null) => Ok(None),
            Some(serde_json::Value::String(s)) => baggage::decode(s),
            Some(_) => Err(ContextError("_meta.baggage must be a string".into())),
        }
    }
}
