//! Dollar costs for a rig agent from its traces alone, offline against rig's
//! mock model.
//!
//! ```text
//! cargo run --example rig_tracing_cost --features tracing-layer,__rig-test-utils
//! ```
//!
//! rig records OpenTelemetry GenAI fields (`gen_ai.request.model`,
//! `gen_ai.usage.input_tokens`, ...) on a span for every model call.
//! `CostLayer` sits in the tracing subscriber, prices each of those spans with
//! llm-prices and emits `gen_ai.usage.cost` inside it. The agent code is plain
//! rig: no hook, no wrapper. With `tracing-opentelemetry` added after the
//! layer, the cost travels with the span to Jaeger, Langfuse, Honeycomb, ...
//!
//! The mock model reports itself as `mock-model`, so a price hook bills it at
//! gpt-4o's rate. With a real provider, drop the hook: the model ids rig
//! records are priced as they are.

use std::sync::Arc;

use llm_budget::tracing_layer::CostLayer;
use llm_budget::{AgentBudget, BudgetError, PriceTable};
use rig_agent::agent::AgentBuilder;
use rig_agent::completion::Usage;
use rig_agent::test_utils::{MockAddTool, MockCompletionModel, MockTurn};
use tracing_subscriber::prelude::*;

/// A tool-calling turn that bills 10,000 prompt + 500 completion tokens.
fn paid_turn(id: &str) -> MockTurn {
    MockTurn::tool_call(id, "add", serde_json::json!({"x": 1, "y": 2}))
        .with_usage(Usage::new().input_tokens(10_000).output_tokens(500))
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), BudgetError> {
    // Bill rig's mock model as gpt-4o ($2.50 / $10 per million tokens).
    let prices = PriceTable::from_llm_prices().with_hook(|model, usage| {
        (model == "mock-model").then(|| PriceTable::builtin().cost_usage("gpt-4o", usage).ok()).flatten()
    });

    // An optional cap: spend is booked here, and going over it kills the
    // budget so anything that checks it before a call (`reserve_for`,
    // `rig::BudgetHook`, your own loop) stops. $0.08 covers two of the three
    // turns below.
    let budget = Arc::new(AgentBudget::builder("traced-agent", 0.08).build()?);

    let cost = CostLayer::new().price_table(prices).budget(budget.clone()).on_cost(|c| {
        println!(
            "  {:<22} {:>6} in / {:>4} out  ${:.4}",
            c.span_name, c.usage.input_tokens, c.usage.output_tokens, c.cost_usd
        )
    });
    let totals = cost.tracker();

    // CostLayer first, then whatever exports your spans (fmt here).
    tracing_subscriber::registry()
        .with(cost)
        .with(
            tracing_subscriber::fmt::layer()
                .with_target(true)
                .with_filter(tracing_subscriber::filter::filter_fn(|m| m.target() == "llm_budget::cost")),
        )
        .init();

    // Plain rig: three paid turns, then an answer.
    let agent = AgentBuilder::new(MockCompletionModel::from_turns([
        paid_turn("a"),
        paid_turn("b"),
        paid_turn("c"),
        MockTurn::text("done"),
    ]))
    .tool(MockAddTool)
    .build();

    match agent.prompt("research the topic").max_turns(10).await {
        Ok(_) => println!("agent finished"),
        Err(e) => println!("agent stopped: {e}"),
    }

    println!(
        "traced spend ${:.4} over {} calls ({} unpriced); budget ${:.4} of ${:.2}, killed: {}",
        totals.total_usd(),
        totals.priced_calls(),
        totals.unpriced_calls(),
        budget.spent_usd(),
        budget.limit_usd(),
        budget.is_killed()
    );
    // Three turns at $0.03 (the answer turn reported no usage). rig's
    // invoke_agent span also carries the run's 30,000 / 1,500 token sum; it
    // is not counted again.
    assert_eq!(totals.priced_calls(), 3);
    assert!((totals.total_usd() - 0.09).abs() < 1e-9);
    // Every paid turn is booked, and the third one tripped the cap.
    assert!((budget.spent_usd() - totals.total_usd()).abs() < 1e-9);
    assert!(budget.is_killed());
    Ok(())
}
