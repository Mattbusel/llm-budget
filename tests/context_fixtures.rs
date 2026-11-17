//! BudgetContext codecs against the golden fixtures shared with the Python
//! package agent-budget (tests/fixtures/budget_context), plus fuzzing of
//! the readers that take untrusted input.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use llm_budget::context::{a2a, baggage, mcp, Amount, BudgetContext, IncurredBooker};
use llm_budget::{AgentBudget, Limit};
use proptest::prelude::*;
use serde_json::{json, Value};

fn fixture(name: &str) -> Value {
    let path = format!("{}/tests/fixtures/budget_context/{name}", env!("CARGO_MANIFEST_DIR"));
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

#[test]
fn roundtrip_fixtures() {
    let f = fixture("roundtrip.json");
    for case in f["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let ctx = BudgetContext::from_json(&case["context"]).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(ctx.to_json().unwrap(), case["context"], "{name}: JSON round trip");
        assert_eq!(baggage::encode(&ctx), case["baggage"].as_str().unwrap(), "{name}: baggage");
        let back = baggage::decode(case["baggage"].as_str().unwrap()).unwrap();
        let want = &case["baggage_context"];
        match back {
            None => assert!(want.is_null(), "{name}: decoded nothing"),
            Some(b) => assert_eq!(&b.to_json().unwrap(), want, "{name}: baggage round trip"),
        }
        // A2A metadata carries the same JSON under the extension URI.
        let md = a2a::to_metadata(&ctx).unwrap();
        assert_eq!(md[a2a::EXTENSION_URI], case["context"]);
        assert_eq!(a2a::from_metadata(&md).unwrap(), Some(ctx));
    }
}

#[test]
fn normalize_fixtures() {
    let f = fixture("normalize.json");
    for case in f["json"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let ctx = BudgetContext::from_json(&case["input"]).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(ctx.to_json().unwrap(), case["context"], "{name}");
    }
    for case in f["baggage"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let got = baggage::decode(case["input"].as_str().unwrap()).unwrap_or_else(|e| panic!("{name}: {e}"));
        let got = got.map(|c| c.to_json().unwrap()).unwrap_or(Value::Null);
        assert_eq!(got, case["context"], "{name}");
    }
}

#[test]
fn invalid_fixtures() {
    let f = fixture("invalid.json");
    for input in f["json"].as_array().unwrap() {
        assert!(BudgetContext::from_json(input).is_err(), "accepted {input}");
        let md = json!({ a2a::EXTENSION_URI: input });
        assert!(a2a::from_metadata(&md).is_err(), "accepted metadata {input}");
    }
    for input in f["baggage"].as_array().unwrap() {
        let s = input.as_str().unwrap();
        assert!(baggage::decode(s).is_err(), "accepted baggage {s:?}");
    }
}

#[test]
fn oversized_baggage_is_refused() {
    let big = format!("llm.budget.tokens.limit=1,pad={}", "x".repeat(baggage::MAX_BYTES));
    assert!(baggage::decode(&big).is_err());
}

#[test]
fn merge_keeps_other_members_and_replaces_ours() {
    let ctx = BudgetContext::from_json(&json!({"budget": {"cost": {"amount": "1", "currency": "USD"}}})).unwrap();
    let merged = baggage::merge(Some("a=1, llm.budget.cost.amount=9 ,b=2;p"), &ctx);
    assert_eq!(merged, "a=1,b=2;p,llm.budget.cost.amount=1,llm.budget.cost.currency=USD");
    let mut meta = serde_json::Map::new();
    meta.insert("baggage".into(), json!("tenant=acme"));
    mcp::inject(&mut meta, &ctx);
    assert_eq!(meta["baggage"], "tenant=acme,llm.budget.cost.amount=1,llm.budget.cost.currency=USD");
    assert_eq!(mcp::extract(&Value::Object(meta)).unwrap(), Some(ctx));
}

#[test]
fn delegation_round_trip_through_a2a_metadata() {
    let parent = AgentBudget::builder("orchestrator", 1.00).build().unwrap();
    let slice = parent.child("remote", Limit::Usd(0.40)).unwrap();
    let out = BudgetContext::outbound(&slice).with_deadline("2026-12-01T00:00:00Z");
    let r = out.reservation.as_ref().unwrap();
    assert!(r.id.starts_with("rsv_") && r.id.len() == 36);
    assert_eq!(r.amount, Amount(400_000_000));
    // Each reservation gets its own id.
    assert_ne!(BudgetContext::outbound(&slice).reservation.unwrap().id, r.id);

    // The callee builds its own budget from what it received, delegates once.
    let md = a2a::to_metadata(&out).unwrap();
    let got = a2a::from_metadata(&md).unwrap().unwrap();
    let callee = AgentBudget::from_context("callee", &got).unwrap();
    assert!((callee.limit_usd() - 0.40).abs() < 1e-12);
    callee.reserve(0.10).unwrap().settle(0.07);
    let sub = callee.child("sub", Limit::Usd(0.10)).unwrap();
    sub.reserve(0.05).unwrap().settle(0.05);
    drop(sub);

    let report = BudgetContext::reporting(&callee);
    let inc = report.incurred.as_ref().unwrap();
    assert_eq!(report.to_json().unwrap()["incurred"]["cost"], json!({"amount": "0.07", "currency": "USD", "source": "local"}));
    assert_eq!(inc.aggregated_child_cost.as_ref().unwrap().amount, Amount(50_000_000));
    assert_eq!(inc.delegations, Some(1));

    // The caller books the reported total once, even if it hears it twice.
    let mut booker = IncurredBooker::new();
    assert!(booker.book(&slice, report.incurred.as_ref()).is_some());
    assert!(booker.book(&slice, report.incurred.as_ref()).is_none());
    assert!((parent.spent_usd() - 0.12).abs() < 1e-12);
    let receipt = slice.settle();
    assert!((receipt.released_usd - 0.28).abs() < 1e-12);
}

#[test]
fn a_capped_booker_books_at_most_the_reservation() {
    let parent = AgentBudget::builder("caller", 1.00).build().unwrap();
    let slice = parent.child("remote", Limit::Usd(0.10)).unwrap();
    let report = BudgetContext::from_json(&json!({"incurred": {"cost": {"amount": "5", "currency": "USD"}}})).unwrap();
    let mut booker = IncurredBooker::capped_at(Amount(100_000_000));
    let out = booker.book(&slice, report.incurred.as_ref()).unwrap();
    assert!(!out.over_cap);
    assert_eq!(booker.booked(), Amount(100_000_000));
    assert_eq!(booker.excess(), Amount(4_900_000_000));
    assert!(booker.book(&slice, report.incurred.as_ref()).is_none());
    drop(slice);
    assert!((parent.spent_usd() - 0.10).abs() < 1e-12);
    assert!(parent.reserved_usd().abs() < 1e-12);
}

#[test]
fn the_largest_report_saturates() {
    let b = AgentBudget::builder("caller", 1.00).build().unwrap();
    let report = BudgetContext::from_json(&json!({"incurred": {
        "cost": {"amount": "18446744073.709551615", "currency": "USD"},
        "aggregatedChildCost": {"amount": "10000000000", "currency": "USD"}
    }}))
    .unwrap();
    let out = IncurredBooker::new().book(&b, report.incurred.as_ref()).unwrap();
    assert!(out.over_cap);
}

#[test]
fn no_usd_amount_is_an_error() {
    let ctx = BudgetContext::from_json(&json!({"budget": {"cost": {"amount": "1", "currency": "EUR"}}})).unwrap();
    assert!(AgentBudget::from_context("x", &ctx).is_err());
    assert!(AgentBudget::from_context("x", &BudgetContext::default()).is_err());
}

fn amount() -> impl Strategy<Value = String> {
    (0u64..1_000_000, 0u32..1_000_000_000).prop_map(|(w, f)| Amount(w * 1_000_000_000 + u64::from(f)).to_string())
}

fn text() -> impl Strategy<Value = String> {
    "[ -~\u{e9}\u{4e2d}]{1,40}"
}

fn ascii() -> impl Strategy<Value = String> {
    "[ -~]{1,64}"
}

fn context() -> impl Strategy<Value = Value> {
    (
        proptest::option::of(amount()),
        proptest::option::of(any::<u32>()),
        proptest::option::of(ascii()),
        proptest::option::of((text(), amount(), proptest::option::of(ascii()))),
    )
        .prop_map(|(cost, limit, deadline, res)| {
            let mut b = serde_json::Map::new();
            if let Some(c) = cost {
                b.insert("cost".into(), json!({"amount": c, "currency": "USD"}));
            }
            if let Some(l) = limit {
                b.insert("tokens".into(), json!({"limit": l}));
            }
            if let Some(d) = deadline {
                b.insert("deadline".into(), json!(d));
            }
            let mut top = serde_json::Map::new();
            if !b.is_empty() {
                top.insert("budget".into(), Value::Object(b));
            }
            if let Some((id, amt, exp)) = res {
                let mut r = json!({"id": id, "amount": amt, "currency": "USD"});
                if let Some(e) = exp {
                    r["expires"] = json!(e);
                }
                top.insert("reservation".into(), r);
            }
            Value::Object(top)
        })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(1024))]

    /// Untrusted headers: decode returns Ok or Err, never panics.
    #[test]
    fn decode_never_panics(s in "\\PC{0,200}") {
        let _ = baggage::decode(&s);
        let _ = baggage::decode(&format!("llm.budget.cost.amount={s},llm.budget.cost.currency=USD"));
        let _ = baggage::decode(&format!("llm.budget.reservation.id={s}"));
    }

    /// Untrusted JSON: from_json returns Ok or Err, never panics.
    #[test]
    fn from_json_never_panics(v in proptest::arbitrary::any::<(String, i64, f64, bool)>()) {
        let (s, i, f, b) = v;
        for amount in [json!(s.clone()), json!(i), json!(f), json!(b)] {
            let _ = BudgetContext::from_json(&json!({"budget": {"cost": {"amount": amount, "currency": "USD"}}}));
            let _ = BudgetContext::from_json(&json!({"incurred": {"delegations": amount, "tokens": {"input": amount}}}));
        }
    }

    /// Every valid outbound context survives JSON and baggage unchanged.
    #[test]
    fn outbound_contexts_round_trip(v in context()) {
        let ctx = BudgetContext::from_json(&v).unwrap();
        prop_assert_eq!(&ctx.to_json().unwrap(), &v);
        let header = baggage::encode(&ctx);
        let back = baggage::decode(&header).unwrap().unwrap_or_default();
        prop_assert_eq!(back, ctx);
    }

    /// Amounts print and parse back exactly.
    #[test]
    fn amounts_round_trip(n in any::<u64>()) {
        prop_assert_eq!(Amount::parse(&Amount(n).to_string()).unwrap(), Amount(n));
    }
}
