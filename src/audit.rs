//! Spend audit: replay a log of LLM calls and show where a soft limit would
//! have let money through that a hard cap refuses.
//!
//! Most budgets in agent frameworks are soft limits: they check what was
//! spent *before* a call, book the cost *after* it, price unknown models at a
//! default (often $0), and price cached tokens as plain input. Each of those
//! lets money past the cap. [`replay`] runs the same calls two ways:
//!
//! - **soft limit**: the common pattern. A call goes through while the
//!   *tracked* spend is under the cap; the tracker books the cost your log
//!   says it booked (`booked_usd`), or the true cost when the log has none,
//!   and $0 for a model it cannot price.
//! - **hard cap, fail closed** (llm-budget's default): a call is admitted only
//!   if its true cost fits in what is left, and refused if its price is
//!   unknown, missing a cache rate, or too old.
//!
//! and reports the true cost of everything the soft limit let through, the
//! call at which real spend crossed the cap, and the money past it.
//!
//! The `llm-budget` binary wraps this: `llm-budget audit calls.jsonl --cap 5`.
//!
//! # Input
//!
//! One JSON object per line, with a `model` and the provider's usage in any
//! shape [`Usage::from_value`] reads (OpenAI, Anthropic or Gemini; the whole
//! response or the usage block, nested under `usage` or at the top level).
//! Optional: `booked_usd` (or `cost_usd`, `cost`), what your current tracker
//! booked for the call. Blank lines are skipped.
//!
//! ```
//! use llm_budget::audit::{parse_jsonl, replay, AuditConfig};
//!
//! let log = r#"
//! {"model": "gpt-4o", "usage": {"prompt_tokens": 100000, "completion_tokens": 2000}, "booked_usd": 0.27}
//! {"model": "gpt-4o", "usage": {"prompt_tokens": 100000, "completion_tokens": 2000}, "booked_usd": 0.27}
//! {"model": "my-finetune", "usage": {"prompt_tokens": 100000, "completion_tokens": 2000}, "booked_usd": 0.0}
//! "#;
//! let calls = parse_jsonl(log)?;
//! let report = replay(&calls, &AuditConfig::new(0.30));
//! // The soft limit checks $0.27 < $0.30 and lets the second call through:
//! // $0.54 spent on a $0.30 cap.
//! assert_eq!(report.soft.calls_through, 2);
//! assert!(report.soft.money_past_cap_usd > 0.23);
//! // The hard cap admits one call and refuses the rest.
//! assert_eq!(report.hard.admitted, 1);
//! assert_eq!(report.hard.refused_unpriced, 1);
//! # Ok::<(), llm_budget::BudgetError>(())
//! ```

use crate::error::BudgetError;
use crate::pricing::PriceTable;
use crate::usage::Usage;
use std::collections::BTreeMap;

/// One call from the log.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct Call {
    /// 1-based line number in the input.
    pub line: usize,
    /// The model id as logged.
    pub model: String,
    /// Normalised token counts.
    pub usage: Usage,
    /// What the existing tracker booked for the call, if the log says.
    pub booked_usd: Option<f64>,
}

/// Parse a JSON-lines log into [`Call`]s.
///
/// # Errors
/// [`BudgetError::InvalidConfig`] naming the first line that is not a JSON
/// object with a string `model` and a usage block.
pub fn parse_jsonl(text: &str) -> Result<Vec<Call>, BudgetError> {
    let mut calls = Vec::new();
    for (i, raw) in text.lines().enumerate() {
        let line = raw.trim();
        if line.is_empty() {
            continue;
        }
        let bad = |why: String| BudgetError::InvalidConfig(format!("line {}: {why}", i + 1));
        let v: serde_json::Value = serde_json::from_str(line).map_err(|e| bad(format!("not JSON: {e}")))?;
        let model = v
            .get("model")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| bad("no string \"model\" field".into()))?
            .to_string();
        let usage = Usage::from_value(&v).map_err(|e| bad(e.to_string()))?;
        let booked_usd = ["booked_usd", "cost_usd", "cost"]
            .iter()
            .find_map(|k| v.get(*k).and_then(serde_json::Value::as_f64))
            .filter(|x| x.is_finite() && *x >= 0.0);
        calls.push(Call { line: i + 1, model, usage, booked_usd });
    }
    Ok(calls)
}

/// What [`replay`] compares against.
#[derive(Debug, Clone)]
pub struct AuditConfig {
    /// The cap, in USD.
    pub cap_usd: f64,
    /// Prices last checked longer ago than this are refused by the hard cap.
    pub max_price_age_days: u32,
    /// Judge price ages as of this day (days since 1970-01-01); `None` = today.
    pub today: Option<i64>,
    /// The prices; `None` = [`PriceTable::builtin`].
    pub prices: Option<PriceTable>,
}

impl AuditConfig {
    /// A cap of `cap_usd` with the built-in prices and the default price age.
    pub fn new(cap_usd: f64) -> Self {
        Self { cap_usd, max_price_age_days: crate::safety::DEFAULT_MAX_PRICE_AGE_DAYS, today: None, prices: None }
    }
}

/// How the soft limit did on the log.
#[derive(Debug, Clone, Default, PartialEq, serde::Serialize)]
pub struct SoftLimitResult {
    /// Calls the soft limit let through.
    pub calls_through: usize,
    /// The 1-based call index at which the soft limit stopped the run, if it did.
    pub fired_at_call: Option<usize>,
    /// What the tracker had booked when it stopped (or at the end).
    pub tracked_usd: f64,
    /// The true cost of the priced calls it let through.
    pub true_usd: f64,
    /// Calls it let through whose true cost is unknown (no price).
    pub unpriced_calls_through: usize,
    /// The 1-based call index at which the true spend first went past the cap.
    pub crossed_cap_at_call: Option<usize>,
    /// Calls let through after the true spend was already past the cap.
    pub calls_past_cap: usize,
    /// `true_usd - cap`, when positive: money spent past the cap. A lower
    /// bound when unpriced calls went through.
    pub money_past_cap_usd: f64,
}

/// How the fail-closed hard cap did on the log.
#[derive(Debug, Clone, Default, PartialEq, serde::Serialize)]
pub struct HardCapResult {
    /// Calls admitted.
    pub admitted: usize,
    /// True cost of the admitted calls (never above the cap).
    pub spent_usd: f64,
    /// Calls refused because they did not fit under the cap.
    pub refused_over_cap: usize,
    /// Calls refused because the model had no price.
    pub refused_unpriced: usize,
    /// Calls refused because the call used the prompt cache and the price has no cache rate.
    pub refused_cache_price: usize,
    /// Calls refused because the price was older than the limit.
    pub refused_stale: usize,
}

/// One model's booked-versus-true totals, for models the log gives a booked cost for.
#[derive(Debug, Clone, Default, PartialEq, serde::Serialize)]
pub struct ModelDrift {
    /// The model id.
    pub model: String,
    /// Calls with both a booked and a true cost.
    pub calls: usize,
    /// What the tracker booked.
    pub booked_usd: f64,
    /// What the calls really cost.
    pub true_usd: f64,
}

impl ModelDrift {
    /// `booked / true`: below 1 the tracker under-books (money leaks past a
    /// cap), above 1 it over-books (budget wasted). `None` when the true cost is 0.
    pub fn ratio(&self) -> Option<f64> {
        (self.true_usd > 0.0).then(|| self.booked_usd / self.true_usd)
    }
}

/// A model that could not be priced, with why.
#[derive(Debug, Clone, Default, PartialEq, serde::Serialize)]
pub struct Unpriced {
    /// The model id.
    pub model: String,
    /// The error that refused it.
    pub reason: String,
    /// How many calls.
    pub calls: usize,
}

/// The result of [`replay`].
#[derive(Debug, Clone, Default, PartialEq, serde::Serialize)]
pub struct AuditReport {
    /// The cap.
    pub cap_usd: f64,
    /// Calls in the log.
    pub calls: usize,
    /// True cost of every priced call in the log.
    pub true_total_usd: f64,
    /// Calls with a usable price.
    pub priced_calls: usize,
    /// The soft-limit replay.
    pub soft: SoftLimitResult,
    /// The fail-closed replay.
    pub hard: HardCapResult,
    /// Booked versus true cost per model, worst ratio first.
    pub drift: Vec<ModelDrift>,
    /// Models with no usable price.
    pub unpriced: Vec<Unpriced>,
    /// Models whose price is older than the limit, with the date checked.
    pub stale: BTreeMap<String, String>,
}

impl AuditReport {
    /// True when the soft limit let money past the cap, or a call had no price.
    pub fn leaked(&self) -> bool {
        self.soft.money_past_cap_usd > 0.0 || self.soft.unpriced_calls_through > 0
    }
}

/// Why one call could not be priced by the hard cap.
enum Refusal {
    Unpriced(String),
    CachePrice(String),
    Stale,
}

/// Replay `calls` under a soft limit and under a fail-closed hard cap.
pub fn replay(calls: &[Call], cfg: &AuditConfig) -> AuditReport {
    let table = cfg.prices.as_ref().unwrap_or_else(|| PriceTable::builtin());
    let today = cfg.today.unwrap_or_else(crate::safety::today_days);
    let mut report = AuditReport { cap_usd: cfg.cap_usd, calls: calls.len(), ..AuditReport::default() };
    let mut unpriced: BTreeMap<String, Unpriced> = BTreeMap::new();
    let mut drift: BTreeMap<String, ModelDrift> = BTreeMap::new();
    let (mut soft, mut hard) = (SoftLimitResult::default(), HardCapResult::default());
    let mut stopped = false;

    for (i, call) in calls.iter().enumerate() {
        let n = i + 1;
        let priced = table.cost_usage(&call.model, &call.usage);
        let true_cost = priced.as_ref().ok().copied();
        if let Some(c) = true_cost {
            report.priced_calls += 1;
            report.true_total_usd += c;
        }
        let refusal = match &priced {
            Ok(_) => table
                .checked_date(&call.model, &call.usage)
                .and_then(|d| Some((d, today - crate::safety::days_from_date(d)?)))
                .filter(|(_, age)| *age > i64::from(cfg.max_price_age_days))
                .map(|(d, _)| {
                    report.stale.insert(call.model.clone(), d.to_string());
                    Refusal::Stale
                }),
            Err(e @ BudgetError::CachePriceMissing { .. }) => Some(Refusal::CachePrice(e.to_string())),
            Err(e) => Some(Refusal::Unpriced(e.to_string())),
        };
        if let Some(Refusal::Unpriced(why) | Refusal::CachePrice(why)) = &refusal {
            let entry = unpriced.entry(call.model.clone()).or_insert_with(|| Unpriced {
                model: call.model.clone(),
                reason: why.clone(),
                calls: 0,
            });
            entry.calls += 1;
        }
        if let (Some(c), Some(b)) = (true_cost, call.booked_usd) {
            let d = drift
                .entry(call.model.clone())
                .or_insert_with(|| ModelDrift { model: call.model.clone(), ..ModelDrift::default() });
            d.calls += 1;
            d.booked_usd += b;
            d.true_usd += c;
        }

        // Soft limit: check the tracked spend before the call, book after.
        if !stopped {
            if soft.tracked_usd >= cfg.cap_usd {
                stopped = true;
                soft.fired_at_call = Some(n);
            } else {
                soft.calls_through += 1;
                if soft.true_usd > cfg.cap_usd {
                    soft.calls_past_cap += 1;
                }
                soft.tracked_usd += call.booked_usd.or(true_cost).unwrap_or(0.0);
                match true_cost {
                    Some(c) => {
                        soft.true_usd += c;
                        if soft.crossed_cap_at_call.is_none() && soft.true_usd > cfg.cap_usd {
                            soft.crossed_cap_at_call = Some(n);
                        }
                    }
                    None => soft.unpriced_calls_through += 1,
                }
            }
        }

        // Hard cap, fail closed: admit only what is priced, fresh and fits.
        match (&refusal, true_cost) {
            (Some(Refusal::Unpriced(_)), _) => hard.refused_unpriced += 1,
            (Some(Refusal::CachePrice(_)), _) => hard.refused_cache_price += 1,
            (Some(Refusal::Stale), _) => hard.refused_stale += 1,
            (None, Some(c)) if hard.spent_usd + c <= cfg.cap_usd + 1e-12 => {
                hard.admitted += 1;
                hard.spent_usd += c;
            }
            (None, _) => hard.refused_over_cap += 1,
        }
    }
    soft.money_past_cap_usd = (soft.true_usd - cfg.cap_usd).max(0.0);
    report.soft = soft;
    report.hard = hard;
    report.unpriced = unpriced.into_values().collect();
    let mut drift: Vec<ModelDrift> =
        drift.into_values().filter(|d| d.ratio().is_some_and(|r| (r - 1.0).abs() > 0.01)).collect();
    drift.sort_by(|a, b| {
        let off = |d: &ModelDrift| d.ratio().map_or(0.0, |r| (r.ln()).abs());
        off(b).partial_cmp(&off(a)).unwrap_or(std::cmp::Ordering::Equal)
    });
    report.drift = drift;
    report
}

impl std::fmt::Display for AuditReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let cap = self.cap_usd;
        writeln!(f, "{} calls, cap ${cap:.2}", self.calls)?;
        writeln!(
            f,
            "True cost of every call: ${:.4} ({} priced, {} without a usable price)",
            self.true_total_usd,
            self.priced_calls,
            self.calls - self.priced_calls
        )?;
        writeln!(f)?;
        let s = &self.soft;
        writeln!(f, "Soft limit (checks spend before each call, books after, $0 for unpriced models):")?;
        match s.fired_at_call {
            Some(n) => writeln!(f, "  stopped the run at call {n}, after letting {} calls through", s.calls_through)?,
            None => writeln!(f, "  never stopped the run: all {} calls went through", s.calls_through)?,
        }
        writeln!(f, "  tracker booked ${:.4}; those calls really cost ${:.4}", s.tracked_usd, s.true_usd)?;
        if s.money_past_cap_usd > 0.0 {
            let n = s.crossed_cap_at_call.map_or_else(|| "?".to_string(), |n| n.to_string());
            writeln!(
                f,
                "  MONEY PAST THE CAP: ${:.4} ({:.2}x the cap); real spend crossed ${cap:.2} at call {n}, then {} more calls ran",
                s.money_past_cap_usd,
                s.true_usd / cap,
                s.calls_past_cap
            )?;
        } else {
            writeln!(f, "  no money past the cap")?;
        }
        if s.unpriced_calls_through > 0 {
            writeln!(
                f,
                "  plus {} calls with no price, booked at $0: their cost is unknown",
                s.unpriced_calls_through
            )?;
        }
        writeln!(f)?;
        let h = &self.hard;
        writeln!(f, "Hard cap, fail closed (llm-budget default):")?;
        writeln!(f, "  admitted {} calls, spent ${:.4} of ${cap:.2}", h.admitted, h.spent_usd)?;
        writeln!(
            f,
            "  refused {} over the cap, {} unpriced, {} missing a cache price, {} with a stale price",
            h.refused_over_cap, h.refused_unpriced, h.refused_cache_price, h.refused_stale
        )?;
        if !self.drift.is_empty() {
            writeln!(f)?;
            writeln!(f, "Booked by your tracker vs true cost:")?;
            for d in self.drift.iter().take(15) {
                writeln!(
                    f,
                    "  {:<32} {:>6} calls  booked ${:>10.4}  true ${:>10.4}  {:>6.2}x",
                    d.model,
                    d.calls,
                    d.booked_usd,
                    d.true_usd,
                    d.ratio().unwrap_or(0.0)
                )?;
            }
        }
        if !self.unpriced.is_empty() {
            writeln!(f)?;
            writeln!(f, "No usable price (a soft limit books these at $0):")?;
            for u in &self.unpriced {
                writeln!(f, "  {} ({} calls): {}", u.model, u.calls, u.reason)?;
            }
        }
        if !self.stale.is_empty() {
            writeln!(f)?;
            writeln!(f, "Prices older than the limit (update llm-prices):")?;
            for (m, d) in &self.stale {
                writeln!(f, "  {m}: last checked {d}")?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn call(model: &str, input: u64, output: u64, booked: Option<f64>) -> Call {
        Call { line: 0, model: model.into(), usage: Usage::new(input, output), booked_usd: booked }
    }

    #[test]
    fn soft_limit_lets_the_last_call_overshoot() {
        // $0.27 per call, cap $0.50: the soft limit sees $0.27 < $0.50 and
        // lets the second call through; the hard cap admits one.
        let c = call("gpt-4o", 100_000, 2_000, None);
        let r = replay(&[c.clone(), c.clone(), c], &AuditConfig::new(0.50));
        assert_eq!(r.soft.calls_through, 2);
        assert_eq!(r.soft.fired_at_call, Some(3));
        assert_eq!(r.soft.crossed_cap_at_call, Some(2));
        assert!((r.soft.money_past_cap_usd - 0.04).abs() < 1e-9, "{}", r.soft.money_past_cap_usd);
        assert_eq!(r.hard.admitted, 1);
        assert_eq!(r.hard.refused_over_cap, 2);
        assert!(r.hard.spent_usd <= 0.50);
        assert!(r.leaked());
    }

    #[test]
    fn under_booking_tracker_runs_far_past_the_cap() {
        // The PraisonAI pattern: a new model booked at a $1/$3 default.
        // gpt-5.5 at 100k in / 5k out really costs more than 5x that.
        let booked = (100_000.0 * 1.0 + 5_000.0 * 3.0) / 1e6;
        let calls: Vec<Call> = (0..20).map(|_| call("gpt-5.5", 100_000, 5_000, Some(booked))).collect();
        let r = replay(&calls, &AuditConfig::new(1.0));
        let truth = PriceTable::builtin().cost("gpt-5.5", 100_000, 5_000).unwrap();
        assert!(truth > 4.0 * booked, "{truth} vs {booked}");
        assert_eq!(r.soft.fired_at_call, Some(10)); // 9 x $0.115 = $1.035
        assert!(r.soft.money_past_cap_usd > 2.0, "{}", r.soft.money_past_cap_usd);
        assert_eq!(r.drift.len(), 1);
        assert!(r.drift[0].ratio().unwrap() < 0.25);
        assert!(r.hard.spent_usd <= 1.0);
        let text = r.to_string();
        assert!(text.contains("MONEY PAST THE CAP"), "{text}");
    }

    #[test]
    fn unpriced_and_cache_missing_calls_are_named() {
        let table = PriceTable::new().with_price(crate::ModelPrice::new("no-cache", "self", 1.0, 1.0));
        let mut cfg = AuditConfig::new(10.0);
        cfg.prices = Some(table);
        let cached = Call {
            line: 1,
            model: "no-cache".into(),
            usage: Usage::new(1_000, 10).with_cache_read(900),
            booked_usd: None,
        };
        let r = replay(&[call("ghost-model", 10, 10, None), cached, call("no-cache", 1_000, 10, None)], &cfg);
        assert_eq!(r.hard.refused_unpriced, 1);
        assert_eq!(r.hard.refused_cache_price, 1);
        assert_eq!(r.hard.admitted, 1);
        assert_eq!(r.soft.unpriced_calls_through, 2);
        assert_eq!(r.unpriced.len(), 2);
        assert!(r.leaked());
    }

    #[test]
    fn stale_prices_are_refused_by_the_hard_cap() {
        let table =
            PriceTable::new().with_price(crate::ModelPrice::new("old", "self", 1.0, 1.0).with_checked("2020-01-01"));
        let mut cfg = AuditConfig::new(10.0);
        cfg.prices = Some(table);
        cfg.today = crate::safety::days_from_date("2026-11-19");
        let r = replay(&[call("old", 1_000, 10, None)], &cfg);
        assert_eq!(r.hard.refused_stale, 1);
        assert_eq!(r.stale.get("old").map(String::as_str), Some("2020-01-01"));
        assert!(!r.leaked());
    }

    #[test]
    fn parses_provider_shapes_and_rejects_bad_lines() {
        let text = concat!(
            r#"{"model":"gpt-4o","usage":{"prompt_tokens":10,"completion_tokens":2},"booked_usd":0.1}"#,
            "\n\n",
            r#"{"model":"claude-sonnet-4-6","input_tokens":5,"output_tokens":1,"cache_read_input_tokens":100}"#,
            "\n"
        );
        let calls = parse_jsonl(text).unwrap();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].booked_usd, Some(0.1));
        assert_eq!(calls[1].line, 3);
        assert_eq!(calls[1].usage.cached_input_tokens, 100);
        assert!(parse_jsonl("{\"usage\":{}}").unwrap_err().to_string().contains("line 1"));
        assert!(parse_jsonl("nope").is_err());
        assert!(parse_jsonl(r#"{"model":"x"}"#).is_err());
    }
}
