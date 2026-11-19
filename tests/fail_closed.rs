//! The 0.10 defaults: every call the budget cannot price exactly is refused,
//! and every opt-out is loud.

use llm_budget::budget::Limit;
use llm_budget::{AgentBudget, BudgetError, ModelPrice, PriceTable, UnsafeEvent, UnsafeKind, Usage};
use std::sync::{Arc, Mutex};

fn collect() -> (Arc<Mutex<Vec<UnsafeEvent>>>, impl Fn(&UnsafeEvent) + Send + Sync + 'static) {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let s = Arc::clone(&seen);
    (seen, move |e: &UnsafeEvent| s.lock().unwrap().push(e.clone()))
}

#[test]
fn a_budget_without_a_hard_cap_does_not_build() {
    for limit in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        let e = AgentBudget::builder("a", limit).build().err().unwrap();
        assert!(matches!(e, BudgetError::InvalidConfig(ref m) if m.contains("hard cap")), "{limit}: {e}");
    }
}

#[test]
fn unknown_models_are_refused_on_every_path() {
    let b = AgentBudget::builder("a", 1.0).build().unwrap();
    let u = Usage::new(1_000, 100);
    assert!(matches!(b.reserve_for("gtp-4o", &u), Err(BudgetError::UnknownModel(_))));
    assert!(matches!(b.record("gtp-4o", &u), Err(BudgetError::UnknownModel(_))));
    let rt = tokio::runtime::Builder::new_current_thread().build().unwrap();
    rt.block_on(async {
        assert!(matches!(b.charge_model("gtp-4o", 1, 1, None, None).await, Err(BudgetError::UnknownModel(_))));
        assert!(matches!(b.charge_usage("gtp-4o", &u, None, None).await, Err(BudgetError::UnknownModel(_))));
    });
    assert_eq!(b.spent_usd(), 0.0);
    assert_eq!(b.reserved_usd(), 0.0);
    assert_eq!(b.unsafe_events(), 0);
}

#[test]
fn cache_tokens_need_an_explicit_cache_price() {
    let table = PriceTable::new()
        .with_price(ModelPrice::new("no-cache", "self", 1.0, 2.0))
        .with_price(ModelPrice::new("read-only", "self", 1.0, 2.0).with_cache(Some(0.1), None));
    let b = AgentBudget::builder("a", 10.0).price_table(table).build().unwrap();
    // No cache tokens: priced.
    assert!(b.reserve_for("no-cache", &Usage::new(1_000, 10)).is_ok());
    // Cache reads with no read price: refused, not billed as plain input.
    let e = b.reserve_for("no-cache", &Usage::new(1_000, 10).with_cache_read(900)).unwrap_err();
    assert_eq!(e, BudgetError::CachePriceMissing { model: "no-cache".into(), kind: "read" });
    // A read price but no write price.
    assert!(b.reserve_for("read-only", &Usage::new(1_000, 10).with_cache_read(900)).is_ok());
    let e = b.record("read-only", &Usage::new(1_000, 10).with_cache_write(500)).unwrap_err();
    assert_eq!(e, BudgetError::CachePriceMissing { model: "read-only".into(), kind: "write" });
    assert!(e.to_string().contains("refusing to guess"));
}

#[test]
fn builtin_cache_prices_are_used_when_listed() {
    // claude-sonnet-4-6 lists cache reads and 5m writes: priced exactly.
    let u = Usage::from_parts(2_048, 45_000, 3_500, 1_024);
    let usd = PriceTable::builtin().cost_usage("claude-sonnet-4-6", &u).unwrap();
    assert!((usd - 0.048_129).abs() < 1e-12, "{usd}");
}

#[test]
fn allow_unpriced_is_loud_on_every_call() {
    let (seen, sink) = collect();
    let b = AgentBudget::builder("loose", 1.0).allow_unpriced().on_unsafe(sink).build().unwrap();
    for _ in 0..3 {
        b.reserve_for("in-house-70b", &Usage::new(1_000, 100)).unwrap().release();
    }
    b.record("in-house-70b", &Usage::new(1, 1)).unwrap();
    assert_eq!(b.unsafe_events(), 4);
    let seen = seen.lock().unwrap();
    assert_eq!(seen.len(), 4);
    assert!(
        matches!(&seen[0].kind, UnsafeKind::UnpricedAllowed { model, held_usd, .. } if model == "in-house-70b" && *held_usd == 0.0)
    );
    assert!(seen[0].to_string().contains("no usable price"));
}

#[test]
fn stale_prices_are_refused_until_updated_or_waived() {
    let u = Usage::new(1_000, 100);
    // Judged far in the future, every built-in price is stale.
    let b = AgentBudget::builder("a", 1.0).price_check_date("2099-01-01").build().unwrap();
    match b.reserve_for("gpt-4o", &u) {
        Err(BudgetError::StalePrice { model, age_days, max_age_days, .. }) => {
            assert_eq!(model, "gpt-4o");
            assert!(age_days > 20_000);
            assert_eq!(max_age_days, llm_budget::DEFAULT_MAX_PRICE_AGE_DAYS);
        }
        other => panic!("expected StalePrice, got {other:?}"),
    }
    let rt = tokio::runtime::Builder::new_current_thread().build().unwrap();
    assert!(matches!(rt.block_on(b.charge_model("gpt-4o", 1, 1, None, None)), Err(BudgetError::StalePrice { .. })));
    // A longer limit.
    let b = AgentBudget::builder("a", 1.0).price_check_date("2099-01-01").max_price_age_days(100_000).build().unwrap();
    assert!(b.reserve_for("gpt-4o", &u).is_ok());
    // The loud opt-out.
    let (seen, sink) = collect();
    let b = AgentBudget::builder("a", 1.0)
        .price_check_date("2099-01-01")
        .allow_stale_prices()
        .on_unsafe(sink)
        .build()
        .unwrap();
    b.reserve_for("gpt-4o", &u).unwrap().release();
    b.reserve_for("gpt-4o", &u).unwrap().release();
    assert_eq!(seen.lock().unwrap().len(), 2);
    assert!(matches!(&seen.lock().unwrap()[0].kind, UnsafeKind::StalePriceUsed { .. }));
}

#[test]
fn todays_builtin_prices_are_fresh() {
    let b = AgentBudget::builder("a", 1.0).build().unwrap();
    assert!(b.reserve_for("claude-sonnet-4-6", &Usage::new(1_000, 100)).is_ok());
    let checked = PriceTable::builtin().checked_date("claude-sonnet-4-6", &Usage::new(1, 1)).unwrap();
    assert!(llm_budget::safety::days_from_date(checked).is_some(), "{checked}");
}

#[test]
fn own_prices_are_judged_only_when_dated() {
    let u = Usage::new(1_000, 100);
    let table = PriceTable::new()
        .with_price(ModelPrice::new("undated", "self", 1.0, 1.0))
        .with_price(ModelPrice::new("old", "self", 1.0, 1.0).with_checked("2020-01-01"));
    let b = AgentBudget::builder("a", 1.0).price_table(table).price_check_date("2026-11-19").build().unwrap();
    assert!(b.reserve_for("undated", &u).is_ok());
    assert!(matches!(b.reserve_for("old", &u), Err(BudgetError::StalePrice { .. })));
}

#[test]
fn a_bad_price_check_date_is_a_config_error() {
    let e = AgentBudget::builder("a", 1.0).price_check_date("19/11/2026").build().err().unwrap();
    assert!(matches!(e, BudgetError::InvalidConfig(_)));
}

#[test]
fn hard_cap_refuses_and_soft_limit_admits_loudly() {
    let hard = AgentBudget::builder("hard", 1.0).build().unwrap();
    hard.reserve(0.9).unwrap().settle(0.9);
    assert!(matches!(hard.reserve(0.2), Err(BudgetError::AgentExhausted { .. })));

    let (seen, sink) = collect();
    let soft = AgentBudget::builder("soft", 1.0).unsafe_soft_limit().on_unsafe(sink).build().unwrap();
    assert!(soft.is_soft_limit());
    soft.reserve(0.9).unwrap().settle(0.9);
    let out = soft.reserve(0.2).unwrap().settle(0.2);
    assert!(out.over_cap);
    assert!((soft.spent_usd() - 1.1).abs() < 1e-9);
    let seen = seen.lock().unwrap();
    assert_eq!(seen.len(), 1);
    match &seen[0].kind {
        UnsafeKind::SoftLimitOverrun { limit_usd, committed_usd } => {
            assert!((limit_usd - 1.0).abs() < 1e-9);
            assert!((committed_usd - 1.1).abs() < 1e-9);
        }
        other => panic!("{other:?}"),
    }
}

#[test]
fn soft_limit_still_stops_on_kill() {
    let soft = AgentBudget::builder("soft", 1.0).unsafe_soft_limit().on_unsafe(|_| {}).build().unwrap();
    soft.kill("stop");
    assert!(matches!(soft.reserve(2.0), Err(BudgetError::KillSwitchActivated { .. })));
}

#[test]
fn soft_limit_charges_are_booked_and_flagged() {
    let (seen, sink) = collect();
    let soft = AgentBudget::builder("soft", 1.0).unsafe_soft_limit().on_unsafe(sink).build().unwrap();
    let rt = tokio::runtime::Builder::new_current_thread().build().unwrap();
    rt.block_on(async {
        soft.charge_usd(0.8, None, None).await.unwrap();
        let r = soft.charge_usd(0.5, None, None).await.unwrap();
        assert!((r.total_spent_usd - 1.3).abs() < 1e-9);
    });
    assert_eq!(soft.reserved_usd(), 0.0);
    assert_eq!(seen.lock().unwrap().len(), 1);
}

#[test]
#[allow(deprecated)]
fn tier_averages_are_refused_by_default() {
    let rt = tokio::runtime::Builder::new_current_thread().build().unwrap();
    let strict = AgentBudget::builder("a", 1.0).build().unwrap();
    let e = rt.block_on(strict.charge(100, 50, None, None)).unwrap_err();
    assert!(matches!(e, BudgetError::Unpriced { ref model, .. } if model.contains("tier average")), "{e}");
    let (seen, sink) = collect();
    let loose = AgentBudget::builder("a", 1.0).allow_unpriced().on_unsafe(sink).build().unwrap();
    rt.block_on(loose.charge(100, 50, None, None)).unwrap();
    assert!(matches!(seen.lock().unwrap()[0].kind, UnsafeKind::TierEstimate { .. }));
}

#[test]
fn child_scopes_share_the_settings_and_the_sink() {
    let (seen, sink) = collect();
    let root = AgentBudget::builder("root", 1.0).allow_unpriced().unsafe_soft_limit().on_unsafe(sink).build().unwrap();
    let child = root.child("worker", Limit::Usd(0.1)).unwrap();
    child.reserve_for("in-house", &Usage::new(1, 1)).unwrap().release();
    child.reserve(0.5).unwrap().settle(0.5);
    assert_eq!(root.unsafe_events(), 2);
    assert_eq!(child.unsafe_events(), 2);
    let seen = seen.lock().unwrap();
    assert_eq!(seen[0].agent_id, "worker");
    assert!(matches!(seen[1].kind, UnsafeKind::SoftLimitOverrun { .. }));
}
