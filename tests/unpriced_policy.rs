//! Every PriceStatus path through reserve_for, settle_usage and record, under
//! each PriceAction.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use llm_budget::{AgentBudget, BudgetError, ModelPrice, PriceAction, PriceStatus, PriceTable, UnpricedPolicy, Usage};

fn near(a: f64, b: f64) -> bool {
    (a - b).abs() < 1e-12
}

fn agent(table: PriceTable, policy: Option<UnpricedPolicy>) -> AgentBudget {
    let b = AgentBudget::builder("a", 1.0).price_table(table);
    match policy {
        Some(p) => b.unpriced_policy(p).build().unwrap(),
        None => b.build().unwrap(),
    }
}

static U: std::sync::LazyLock<Usage> = std::sync::LazyLock::new(|| Usage::new(1_000, 200));

/// A table for each status: (table, model, expected status).
fn cases() -> Vec<(PriceTable, &'static str, PriceStatus)> {
    vec![
        (PriceTable::builtin().clone(), "gpt-4o", PriceStatus::Exact),
        (PriceTable::builtin().clone().with_unpriced("in-house-70b"), "in-house-70b", PriceStatus::Unpriced),
        (PriceTable::builtin().clone(), "no-such-model-q", PriceStatus::Missing),
        (PriceTable::new(), "gpt-4o", PriceStatus::NoCatalog),
    ]
}

#[test]
fn lookup_reports_each_status() {
    for (table, model, want) in cases() {
        let got = table.lookup(model, &U);
        assert_eq!(got.status, want, "{model}");
        assert_eq!(got.cost_usd.is_some(), want == PriceStatus::Exact);
    }
    // A table with only your own rows: a model not in it is Missing, not NoCatalog.
    let own = PriceTable::new().with_price(ModelPrice::new("mine", "self", 1.0, 1.0));
    assert_eq!(own.lookup("other", &U).status, PriceStatus::Missing);
    // A provider prefix on a declared model still counts as declared.
    let t = PriceTable::new().with_unpriced("llama-3-70b");
    assert_eq!(t.lookup("vllm/llama-3-70b", &U).status, PriceStatus::Unpriced);
    // An inserted price beats the unpriced declaration.
    let t = t.with_price(ModelPrice::new("llama-3-70b", "self", 0.5, 0.5));
    assert_eq!(t.lookup("llama-3-70b", &U).status, PriceStatus::Exact);
}

#[test]
fn llm_prices_rows_without_an_input_price_are_unpriced_not_missing() {
    // Rows llm-prices lists with no input price (if this release has any)
    // must read as Unpriced: the catalog knows them.
    let t = PriceTable::builtin();
    for p in llm_prices::all().iter().filter(|p| p.input.is_none()) {
        let s = t.lookup(p.model, &Usage::new(1_000, 0)).status;
        assert!(matches!(s, PriceStatus::Unpriced | PriceStatus::Exact), "{}: {s}", p.model);
    }
}

#[test]
fn without_a_policy_unpriced_models_keep_the_old_error() {
    for (table, model, status) in cases() {
        let a = agent(table, None);
        let r = a.reserve_for(model, &U);
        match status {
            PriceStatus::Exact => assert!(r.is_ok()),
            _ => assert_eq!(r.unwrap_err(), BudgetError::UnknownModel(model.to_string())),
        }
    }
}

#[test]
fn allow_holds_nothing_and_books_zero() {
    let policy = UnpricedPolicy::new(PriceAction::Allow, PriceAction::Allow, PriceAction::Allow);
    for (table, model, status) in cases() {
        let a = agent(table, Some(policy));
        let hold = a.reserve_for(model, &U).unwrap();
        assert_eq!(hold.price_status(), status);
        if status != PriceStatus::Exact {
            assert!(near(hold.estimate_usd(), 0.0));
            let out = hold.settle_usage(model, &U);
            assert!(near(out.recorded_usd, 0.0));
            let out = a.record(model, &U).unwrap();
            assert!(near(out.recorded_usd, 0.0));
        }
    }
}

#[test]
fn reserve_holds_and_books_the_flat_amount() {
    let policy =
        UnpricedPolicy::new(PriceAction::Reserve(0.01), PriceAction::Reserve(0.02), PriceAction::Reserve(0.03));
    let want = [0.0, 0.01, 0.02, 0.03];
    for ((table, model, status), flat) in cases().into_iter().zip(want) {
        let a = agent(table, Some(policy));
        let hold = a.reserve_for(model, &U).unwrap();
        assert_eq!(hold.price_status(), status);
        if status == PriceStatus::Exact {
            continue;
        }
        assert!(near(hold.estimate_usd(), flat), "{status}");
        assert!(near(a.reserved_usd(), flat));
        // The settlement cannot price it either, so the flat amount is booked.
        let out = hold.settle_usage(model, &U);
        assert!(near(out.recorded_usd, flat));
        // record() after the fact books the same flat amount.
        let out = a.record(model, &U).unwrap();
        assert!(near(out.recorded_usd, flat));
        assert!(near(a.spent_usd(), 2.0 * flat));
    }
}

#[test]
fn reserve_is_still_capped() {
    let policy = UnpricedPolicy::new(PriceAction::Reserve(2.0), PriceAction::Deny, PriceAction::Deny);
    let a = agent(PriceTable::new().with_unpriced("big"), Some(policy));
    assert!(matches!(a.reserve_for("big", &U), Err(BudgetError::AgentExhausted { .. })));
}

#[test]
fn deny_refuses_with_the_status_and_books_nothing() {
    let policy = UnpricedPolicy::new(PriceAction::Deny, PriceAction::Deny, PriceAction::Deny);
    for (table, model, status) in cases() {
        let a = agent(table, Some(policy));
        if status == PriceStatus::Exact {
            assert!(a.reserve_for(model, &U).is_ok());
            continue;
        }
        let e = a.reserve_for(model, &U).unwrap_err();
        assert_eq!(e, BudgetError::Unpriced { model: model.to_string(), status });
        assert!(e.to_string().contains(status.as_str()));
        assert_eq!(a.record(model, &U).unwrap_err(), BudgetError::Unpriced { model: model.to_string(), status });
        assert!(near(a.spent_usd(), 0.0));
        assert!(near(a.reserved_usd(), 0.0));
    }
}

#[test]
fn mixed_policy_treats_each_status_on_its_own() {
    let policy = UnpricedPolicy::new(PriceAction::Allow, PriceAction::Deny, PriceAction::Reserve(0.05));
    let a = agent(PriceTable::builtin().clone().with_unpriced("self-hosted"), Some(policy));
    assert!(a.reserve_for("self-hosted", &U).is_ok());
    assert!(matches!(a.reserve_for("typo-modle", &U), Err(BudgetError::Unpriced { status: PriceStatus::Missing, .. })));
}

#[test]
fn invalid_reserve_amounts_are_rejected_at_build() {
    for bad in [-0.01, f64::NAN, f64::INFINITY] {
        let policy = UnpricedPolicy::new(PriceAction::Reserve(bad), PriceAction::Deny, PriceAction::Deny);
        let r = AgentBudget::builder("a", 1.0).unpriced_policy(policy).build();
        assert!(matches!(r, Err(BudgetError::InvalidConfig(_))), "{bad}");
    }
}

#[test]
fn statuses_serialize_like_agentgateway() {
    let names: Vec<String> = [PriceStatus::Exact, PriceStatus::Unpriced, PriceStatus::Missing, PriceStatus::NoCatalog]
        .iter()
        .map(|s| serde_json::to_string(s).unwrap())
        .collect();
    assert_eq!(names, ["\"exact\"", "\"unpriced\"", "\"missing\"", "\"noCatalog\""]);
    let p: UnpricedPolicy =
        serde_json::from_str(r#"{"unpriced":"allow","missing":"deny","noCatalog":{"reserve":0.5}}"#).unwrap();
    assert_eq!(p.no_catalog, PriceAction::Reserve(0.5));
}
