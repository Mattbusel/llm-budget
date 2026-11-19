//! Fail-closed defaults, and the loud ways out of them.
//!
//! Since 0.10 an [`AgentBudget`](crate::AgentBudget) refuses, by default,
//! every call it cannot price exactly:
//!
//! | Situation | Default | Opt-out (warns on every call) |
//! |---|---|---|
//! | Model with no price | [`BudgetError::UnknownModel`](crate::BudgetError::UnknownModel) | [`allow_unpriced`](crate::AgentBudgetBuilder::allow_unpriced) or an [`UnpricedPolicy`](crate::UnpricedPolicy) |
//! | Cache tokens on a price with no cache rate | [`BudgetError::CachePriceMissing`](crate::BudgetError::CachePriceMissing) | the same |
//! | Price older than [`DEFAULT_MAX_PRICE_AGE_DAYS`] | [`BudgetError::StalePrice`](crate::BudgetError::StalePrice) | [`allow_stale_prices`](crate::AgentBudgetBuilder::allow_stale_prices), or a longer [`max_price_age_days`](crate::AgentBudgetBuilder::max_price_age_days) |
//! | Call that does not fit under the cap | refused before it is sent | [`unsafe_soft_limit`](crate::AgentBudgetBuilder::unsafe_soft_limit) |
//! | [`charge`](crate::AgentBudget::charge) priced at a tier average | refused | [`allow_unpriced`](crate::AgentBudgetBuilder::allow_unpriced) |
//!
//! A budget always has a hard cap: [`AgentBudget::builder`](crate::AgentBudget::builder)
//! takes one, and a zero, negative, NaN or infinite cap is a configuration
//! error. Every time an opt-out lets a call through that the default would
//! have refused, the budget emits an [`UnsafeEvent`]: to the sink set with
//! [`on_unsafe`](crate::AgentBudgetBuilder::on_unsafe), or to standard error
//! when none is set, and as a `tracing` warning with the `tracing` feature.
//! [`AgentBudget::unsafe_events`](crate::AgentBudget::unsafe_events) counts them.
//!
//! ```
//! use llm_budget::{AgentBudget, BudgetError, Usage};
//! use std::sync::atomic::{AtomicU64, Ordering};
//! use std::sync::Arc;
//!
//! # fn main() -> Result<(), BudgetError> {
//! // The default: an id nobody prices is refused before the call.
//! let strict = AgentBudget::builder("strict", 1.0).build()?;
//! assert!(matches!(strict.reserve_for("gtp-4o", &Usage::new(1_000, 100)), Err(BudgetError::UnknownModel(_))));
//!
//! // The loud way out: let it through, and hear about it every time.
//! let seen = Arc::new(AtomicU64::new(0));
//! let s = Arc::clone(&seen);
//! let loose = AgentBudget::builder("loose", 1.0)
//!     .allow_unpriced()
//!     .on_unsafe(move |_event| { s.fetch_add(1, Ordering::SeqCst); })
//!     .build()?;
//! loose.reserve_for("gtp-4o", &Usage::new(1_000, 100))?.release();
//! loose.reserve_for("gtp-4o", &Usage::new(1_000, 100))?.release();
//! assert_eq!(seen.load(Ordering::SeqCst), 2);
//! assert_eq!(loose.unsafe_events(), 2);
//! # Ok(())
//! # }
//! ```

use crate::model::ModelTier;
use crate::pricing::PriceStatus;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// How old (in days since it was last checked against its source) a price
/// may be before an [`AgentBudget`](crate::AgentBudget) refuses to use it.
pub const DEFAULT_MAX_PRICE_AGE_DAYS: u32 = 90;

/// Receives every [`UnsafeEvent`]; see
/// [`AgentBudgetBuilder::on_unsafe`](crate::AgentBudgetBuilder::on_unsafe).
pub type UnsafeSink = Arc<dyn Fn(&UnsafeEvent) + Send + Sync>;

/// A call the fail-closed default would have refused, let through by an
/// explicit opt-out.
#[derive(Debug, Clone, PartialEq)]
pub struct UnsafeEvent {
    /// The budget (or child scope) the call ran on.
    pub agent_id: String,
    /// What was let through.
    pub kind: UnsafeKind,
}

/// What an [`UnsafeEvent`] let through.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum UnsafeKind {
    /// [`unsafe_soft_limit`](crate::AgentBudgetBuilder::unsafe_soft_limit)
    /// admitted a call that did not fit under the cap.
    SoftLimitOverrun {
        /// The cap.
        limit_usd: f64,
        /// Spend plus holds after the call was admitted.
        committed_usd: f64,
    },
    /// A model with no usable price was let through by the unpriced-model policy.
    UnpricedAllowed {
        /// The model.
        model: String,
        /// Why it had no price.
        status: PriceStatus,
        /// What was held or booked for it instead of its real price.
        held_usd: f64,
    },
    /// A price older than the budget's limit was used because of
    /// [`allow_stale_prices`](crate::AgentBudgetBuilder::allow_stale_prices).
    StalePriceUsed {
        /// The model.
        model: String,
        /// When its price was last checked (`YYYY-MM-DD`).
        checked: String,
        /// How many days ago that was.
        age_days: i64,
    },
    /// [`AgentBudget::charge`](crate::AgentBudget::charge) booked a tier
    /// average instead of a model's price.
    TierEstimate {
        /// The tier whose average was used.
        tier: ModelTier,
        /// The amount charged.
        cost_usd: f64,
    },
}

impl std::fmt::Display for UnsafeEvent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "budget '{}': ", self.agent_id)?;
        match &self.kind {
            UnsafeKind::SoftLimitOverrun { limit_usd, committed_usd } => write!(
                f,
                "soft limit let a call through over the ${limit_usd:.6} cap (committed ${committed_usd:.6}, ${:.6} over)",
                committed_usd - limit_usd
            ),
            UnsafeKind::UnpricedAllowed { model, status, held_usd } => write!(
                f,
                "model '{model}' has no usable price ({status}); let through at ${held_usd:.6} instead of its real cost"
            ),
            UnsafeKind::StalePriceUsed { model, checked, age_days } => {
                write!(f, "price of '{model}' was last checked {checked} ({age_days} days ago); used anyway")
            }
            UnsafeKind::TierEstimate { tier, cost_usd } => {
                write!(f, "charged a {tier:?}-tier average (${cost_usd:.6}) instead of a model's real price")
            }
        }
    }
}

/// The fail-closed settings of one budget tree, shared by its child scopes.
pub(crate) struct Safety {
    pub(crate) soft_limit: bool,
    pub(crate) allow_stale: bool,
    pub(crate) max_age_days: u32,
    /// The day prices are judged on (days since 1970-01-01); `None` = today.
    pub(crate) today: Option<i64>,
    pub(crate) sink: Option<UnsafeSink>,
    pub(crate) events: AtomicU64,
}

impl Default for Safety {
    fn default() -> Self {
        Self {
            soft_limit: false,
            allow_stale: false,
            max_age_days: DEFAULT_MAX_PRICE_AGE_DAYS,
            today: None,
            sink: None,
            events: AtomicU64::new(0),
        }
    }
}

impl std::fmt::Debug for Safety {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Safety")
            .field("soft_limit", &self.soft_limit)
            .field("allow_stale", &self.allow_stale)
            .field("max_age_days", &self.max_age_days)
            .field("today", &self.today)
            .field("sink", &self.sink.is_some())
            .field("events", &self.events.load(Ordering::SeqCst))
            .finish()
    }
}

impl Safety {
    /// Count, log and deliver one event.
    pub(crate) fn emit(&self, event: UnsafeEvent) {
        self.events.fetch_add(1, Ordering::SeqCst);
        #[cfg(feature = "tracing")]
        tracing::warn!(target: "llm_budget", agent = %event.agent_id, "UNSAFE: {}", event);
        match &self.sink {
            Some(sink) => sink(&event),
            None => eprintln!("llm-budget UNSAFE: {event}"),
        }
    }

    /// How many days old a price checked on `checked` is today.
    pub(crate) fn age_days(&self, checked: &str) -> Option<i64> {
        let today = self.today.unwrap_or_else(today_days);
        Some(today - days_from_date(checked)?)
    }
}

/// Days since 1970-01-01 for a `YYYY-MM-DD` date, or `None` if it is not one.
///
/// ```
/// assert_eq!(llm_budget::safety::days_from_date("1970-01-02"), Some(1));
/// assert_eq!(llm_budget::safety::days_from_date("2026-11-19"), Some(20_776));
/// assert_eq!(llm_budget::safety::days_from_date("2026-13-01"), None);
/// ```
pub fn days_from_date(date: &str) -> Option<i64> {
    let b = date.as_bytes();
    if b.len() != 10 || b[4] != b'-' || b[7] != b'-' {
        return None;
    }
    let num = |s: &str| -> Option<i64> {
        if s.bytes().all(|c| c.is_ascii_digit()) {
            s.parse().ok()
        } else {
            None
        }
    };
    let (y, m, d) = (num(date.get(0..4)?)?, num(date.get(5..7)?)?, num(date.get(8..10)?)?);
    let leap = (y % 4 == 0 && y % 100 != 0) || y % 400 == 0;
    let month_days = [31, if leap { 29 } else { 28 }, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
    if !(1..=12).contains(&m) || d < 1 || d > month_days[(m - 1) as usize] {
        return None;
    }
    // Howard Hinnant's days_from_civil.
    let y = if m <= 2 { y - 1 } else { y };
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let mp = (m + 9) % 12;
    let doy = (153 * mp + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    Some(era * 146_097 + doe - 719_468)
}

/// Today (UTC) in days since 1970-01-01.
pub fn today_days() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| (d.as_secs() / 86_400) as i64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dates_round_trip_known_days() {
        assert_eq!(days_from_date("1970-01-01"), Some(0));
        assert_eq!(days_from_date("2000-03-01"), Some(11_017));
        assert_eq!(days_from_date("2024-02-29"), Some(19_782));
        assert_eq!(days_from_date("2023-02-29"), None);
        assert_eq!(days_from_date("2026-1-01"), None);
        assert_eq!(days_from_date("2026-01-0x"), None);
        assert_eq!(days_from_date("+026-01-01"), None);
        assert_eq!(days_from_date(""), None);
        let a = days_from_date("2026-08-21").unwrap();
        let b = days_from_date("2026-11-19").unwrap();
        assert_eq!(b - a, 90);
    }

    #[test]
    fn event_messages_name_what_was_let_through() {
        let e = UnsafeEvent {
            agent_id: "a".into(),
            kind: UnsafeKind::SoftLimitOverrun { limit_usd: 1.0, committed_usd: 1.25 },
        };
        assert!(e.to_string().contains("0.250000 over"), "{e}");
        let e = UnsafeEvent {
            agent_id: "a".into(),
            kind: UnsafeKind::UnpricedAllowed { model: "m".into(), status: PriceStatus::Missing, held_usd: 0.0 },
        };
        assert!(e.to_string().contains("no usable price (missing)"), "{e}");
    }
}
