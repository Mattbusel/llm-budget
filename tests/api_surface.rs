//! The integration surface other projects build on (docs/INTEGRATING.md).
//!
//! These tests fail to compile if a type downstream code shares across
//! threads or tasks stops being `Send + Sync`, if `BudgetError` stops being a
//! plain `std::error::Error` that `?` can box, or if the five-line
//! integration from the docs stops working without default features.

use llm_budget::{
    AgentBudget, BudgetError, BudgetSnapshot, ChargeOutcome, ChargeReceipt, FleetAllocator, Limit, ModelPrice,
    PriceTable, Reservation, ScopeGuard, ScopeReceipt, SpendRecord, UnpricedPolicy, Usage,
};

fn send_sync_static<T: Send + Sync + 'static>() {}

#[test]
fn shared_types_are_send_sync_static() {
    send_sync_static::<AgentBudget>();
    send_sync_static::<Reservation>();
    send_sync_static::<ScopeGuard>();
    send_sync_static::<FleetAllocator>();
    send_sync_static::<PriceTable>();
    send_sync_static::<ModelPrice>();
    send_sync_static::<UnpricedPolicy>();
    send_sync_static::<Usage>();
    send_sync_static::<BudgetError>();
    send_sync_static::<ChargeOutcome>();
    send_sync_static::<ChargeReceipt>();
    send_sync_static::<ScopeReceipt>();
    send_sync_static::<SpendRecord>();
    send_sync_static::<BudgetSnapshot>();
    send_sync_static::<llm_budget::context::BudgetContext>();
}

#[test]
fn budget_error_boxes_into_std_error() {
    fn boxed() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let agent = AgentBudget::builder("a", 0.01).build()?;
        let hold = agent.reserve(1.0)?; // over the cap: the ? converts BudgetError
        hold.release();
        Ok(())
    }
    let err = boxed().expect_err("a $1 hold cannot fit under a $0.01 cap");
    assert!(err.downcast_ref::<BudgetError>().is_some());
}

/// The five-line integration in docs/INTEGRATING.md, verbatim apart from the
/// stand-in provider call. Synchronous: no async runtime needed.
#[test]
fn five_line_integration() -> Result<(), BudgetError> {
    let call_model = |_prompt: &str| Usage::new(1_800, 350); // any client; returns reported usage

    let budget = AgentBudget::builder("my-agent", 5.00).build()?; // $5 hard cap
    let hold = budget.reserve_for("gpt-4o", &Usage::new(2_000, 500))?; // refused if it would not fit
    let usage = call_model("..."); // the real call, with any client
    hold.settle_usage("gpt-4o", &usage); // books exactly what the provider reported

    assert!(budget.spent_usd() > 0.0);
    assert!(budget.remaining_usd() < 5.0);
    assert_eq!(budget.reserved_usd(), 0.0);
    Ok(())
}

/// The sub-agent pattern from docs/INTEGRATING.md.
#[test]
fn sub_agent_integration() -> Result<(), BudgetError> {
    let parent = AgentBudget::builder("orchestrator", 1.00).build()?;
    let child = parent.child("researcher", Limit::Usd(0.25))?; // reserved from the parent up front
    child.reserve(0.10)?.settle(0.08);
    let receipt = child.settle(); // unused $0.17 goes back to the parent
    assert!((receipt.spent_usd - 0.08).abs() < 1e-9);
    assert!((parent.remaining_usd() - 0.92).abs() < 1e-9);
    Ok(())
}
