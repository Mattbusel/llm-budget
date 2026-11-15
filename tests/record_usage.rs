//! The framework-agnostic reporting API: `record`, `reserve_for` /
//! `settle_usage`, per-agent price tables. Every test here is plain sync code:
//! none of it needs an async runtime.

#![allow(clippy::unwrap_used)]

use llm_budget::{AgentBudget, BudgetError, ChargeContext, ModelPrice, PriceTable, Usage};
use std::sync::Arc;

fn price(model: &str, usage: &Usage) -> f64 {
    PriceTable::builtin().cost_usage(model, usage).unwrap()
}

#[test]
fn record_books_reported_usage_with_cache_prices() {
    let agent = AgentBudget::builder("a", 1.0).build().unwrap();
    let u = Usage::from_parts(2_048, 45_000, 3_500, 1_024);
    let out = agent.record("claude-sonnet-4-6", &u).unwrap();
    assert!((out.recorded_usd - 0.048_129).abs() < 1e-9);
    assert!(!out.over_cap);
    let log = agent.drain_audit_log();
    assert_eq!(log[0].model.as_deref(), Some("claude-sonnet-4-6"));
    assert_eq!(log[0].input_tokens, 2_048 + 45_000 + 3_500);
    assert_eq!(log[0].output_tokens, 1_024);
}

#[test]
fn record_books_past_the_cap_and_flags_it() {
    let agent = AgentBudget::builder("a", 0.01).build().unwrap();
    let out = agent.record("gpt-4o", &Usage::new(1_000_000, 0)).unwrap();
    assert!(out.over_cap);
    assert!((agent.spent_usd() - 2.5).abs() < 1e-9);
    // later reservations are refused
    assert!(matches!(agent.reserve_for("gpt-4o", &Usage::new(1, 1)), Err(BudgetError::AgentExhausted { .. })));
}

#[test]
fn record_unknown_model_books_nothing() {
    let agent = AgentBudget::builder("a", 1.0).build().unwrap();
    let err = agent.record("no-such-model", &Usage::new(10, 10)).unwrap_err();
    assert_eq!(err, BudgetError::UnknownModel("no-such-model".into()));
    assert_eq!(agent.spent_usd(), 0.0);
    assert!(agent.drain_audit_log().is_empty());
}

#[test]
fn record_with_keeps_caller_context() {
    let agent = AgentBudget::builder("a", 1.0).build().unwrap();
    agent
        .record_with("gpt-4o", &Usage::new(100, 10), ChargeContext::new().task("t-1").model("gpt-4o-2024-08-06").tokens(1, 2))
        .unwrap();
    let r = &agent.drain_audit_log()[0];
    assert_eq!(r.task_id.as_deref(), Some("t-1"));
    assert_eq!(r.model.as_deref(), Some("gpt-4o-2024-08-06"));
    assert_eq!((r.input_tokens, r.output_tokens), (1, 2));
}

#[test]
fn reserve_for_then_settle_usage() {
    let agent = AgentBudget::builder("a", 0.10).build().unwrap();
    let estimate = Usage::new(8_000, 1_000);
    let hold = agent.reserve_for("gpt-4o", &estimate).unwrap();
    assert_eq!(hold.model(), Some("gpt-4o"));
    assert!((agent.reserved_usd() - price("gpt-4o", &estimate)).abs() < 1e-9);
    let actual = Usage::new(7_412, 230).with_cache_read(4_096);
    let out = hold.settle_usage("gpt-4o", &actual);
    assert!((out.recorded_usd - price("gpt-4o", &actual)).abs() < 1e-9);
    assert_eq!(agent.reserved_usd(), 0.0);
    assert!((agent.spent_usd() - out.recorded_usd).abs() < 1e-12);
}

#[test]
fn settle_usage_falls_back_to_reserved_model_then_estimate() {
    let agent = AgentBudget::builder("a", 1.0).build().unwrap();
    // The provider echoes an id the table does not know: priced as the reserved model.
    let hold = agent.reserve_for("gpt-4o-mini", &Usage::new(1_000, 1_000)).unwrap();
    let u = Usage::new(1_000_000, 0);
    let out = hold.settle_usage("gpt-4o-mini-2099-12-31", &u);
    assert!((out.recorded_usd - 0.15).abs() < 1e-9);
    assert_eq!(agent.drain_audit_log()[0].model.as_deref(), Some("gpt-4o-mini"));

    // A plain dollar reservation and an unknown model: the estimate is booked.
    let hold = agent.reserve(0.02).unwrap();
    assert_eq!(hold.model(), None);
    let out = hold.settle_usage("mystery", &u);
    assert!((out.recorded_usd - 0.02).abs() < 1e-12);
    assert_eq!(agent.drain_audit_log()[0].model.as_deref(), Some("mystery"));
}

#[test]
fn reserve_for_refuses_unknown_models_and_overruns() {
    let agent = AgentBudget::builder("a", 0.001).build().unwrap();
    assert!(matches!(agent.reserve_for("nope", &Usage::new(1, 1)), Err(BudgetError::UnknownModel(_))));
    assert!(matches!(
        agent.reserve_for("gpt-4o", &Usage::new(1_000_000, 0)),
        Err(BudgetError::AgentExhausted { .. })
    ));
    agent.kill("done");
    assert!(matches!(agent.reserve_for("gpt-4o", &Usage::new(1, 0)), Err(BudgetError::KillSwitchActivated { .. })));
}

#[test]
fn dropped_reservation_books_nothing() {
    let agent = AgentBudget::builder("a", 1.0).build().unwrap();
    {
        let _hold = agent.reserve_for("gpt-4o", &Usage::new(10_000, 1_000)).unwrap();
        assert!(agent.reserved_usd() > 0.0);
    }
    assert_eq!(agent.reserved_usd(), 0.0);
    assert_eq!(agent.spent_usd(), 0.0);
}

#[test]
fn per_agent_price_table_overrides_and_is_shared() {
    let table = Arc::new(
        PriceTable::builtin()
            .clone()
            .with_price(ModelPrice::new("gpt-4o", "openai", 1.0, 1.0)) // negotiated
            .with_price(ModelPrice::new("local-llama", "self", 0.0, 0.0))
            .with_hook(|m, u| (m == "metered").then(|| u.total_tokens() as f64 * 1e-6)),
    );
    let a = AgentBudget::builder("a", 10.0).price_table(Arc::clone(&table)).build().unwrap();
    let b = AgentBudget::builder("b", 10.0).price_table(table).build().unwrap();
    assert!((a.record("gpt-4o", &Usage::new(1_000_000, 0)).unwrap().recorded_usd - 1.0).abs() < 1e-9);
    assert_eq!(b.record("local-llama", &Usage::new(1_000_000, 1_000_000)).unwrap().recorded_usd, 0.0);
    assert!((b.cost_of("metered", &Usage::new(500, 500)).unwrap() - 0.001).abs() < 1e-12);
    // the default agent still uses list prices
    let c = AgentBudget::builder("c", 10.0).build().unwrap();
    assert!((c.cost_of("gpt-4o", &Usage::new(1_000_000, 0)).unwrap() - 2.5).abs() < 1e-9);
    assert!(a.price_table().contains("local-llama"));
    assert!(!c.price_table().contains("local-llama"));
}

#[test]
fn charge_methods_use_the_agent_table() {
    let table = PriceTable::new().with_price(ModelPrice::new("only-model", "x", 1.0, 2.0));
    let agent = AgentBudget::builder("a", 10.0).price_table(table).build().unwrap();
    let rt = tokio::runtime::Builder::new_current_thread().build().unwrap();
    rt.block_on(async {
        let r = agent.charge_model("only-model", 1_000_000, 0, None, None).await.unwrap();
        assert!((r.cost_usd - 1.0).abs() < 1e-9);
        let r = agent.charge_usage("only-model", &Usage::new(0, 1_000_000), None, None).await.unwrap();
        assert!((r.cost_usd - 2.0).abs() < 1e-9);
        // gpt-4o is not in this agent's table
        assert!(matches!(
            agent.charge_model("gpt-4o", 1, 1, None, None).await,
            Err(BudgetError::UnknownModel(_))
        ));
    });
}

#[test]
fn long_context_prices_reach_the_budget() {
    let agent = AgentBudget::builder("a", 10.0).build().unwrap();
    let out = agent.record("gpt-5.4", &Usage::new(300_000, 0)).unwrap();
    assert!((out.recorded_usd - 1.5).abs() < 1e-9); // $5/M above 272k, not $2.50
}

#[test]
fn usage_totals_across_turns() {
    let agent = AgentBudget::builder("a", 10.0).build().unwrap();
    let turns = [Usage::new(1_000, 100), Usage::new(2_000, 200).with_cache_read(1_000)];
    for t in &turns {
        agent.record("gpt-4o", t).unwrap();
    }
    let total: Usage = turns.iter().copied().sum();
    assert_eq!(total.total_tokens(), 3_300);
    let sum: f64 = turns.iter().map(|t| price("gpt-4o", t)).sum();
    assert!((agent.spent_usd() - sum).abs() < 1e-9);
}
