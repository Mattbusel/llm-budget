//! Price the token counters a framework already emits, with no change to the framework.
//!
//! swiftide (feature `metrics`) counts tokens per model through the metrics facade. Wrapping the
//! recorder in `CostLayer::swiftide()` adds `swiftide.usage.cost_usd` next to those counters and
//! books every call on a budget. Here the counters are emitted by hand, exactly as swiftide's
//! `emit_usage` does, so the example runs offline:
//!
//! ```text
//! cargo run --example metrics_cost --features metrics-layer
//! ```

use std::sync::Arc;

use llm_budget::metrics_layer::CostLayer;
use llm_budget::{AgentBudget, BudgetError};
use metrics_util::debugging::{DebugValue, DebuggingRecorder};
use metrics_util::layers::Layer as _;

/// What swiftide-core's `metrics::emit_usage` records after each model call.
fn emit_usage(model: &str, prompt_tokens: u64, completion_tokens: u64) {
    let labels = [("model", model.to_string()), ("pipeline", "docs".to_string())];
    metrics::counter!("swiftide.usage.prompt_tokens", &labels).increment(prompt_tokens);
    metrics::counter!("swiftide.usage.completion_tokens", &labels).increment(completion_tokens);
    metrics::counter!("swiftide.usage.total_tokens", &labels).increment(prompt_tokens + completion_tokens);
}

fn main() -> Result<(), BudgetError> {
    // A $0.20 cap for the whole indexing run
    let budget = Arc::new(AgentBudget::builder("index-docs", 0.20).build()?);

    // In an app: CostLayer::swiftide().layer(PrometheusBuilder::new().build_recorder()), then
    // metrics::set_global_recorder(..). A debugging recorder lets us print what was recorded.
    let inner = DebuggingRecorder::new();
    let snapshotter = inner.snapshotter();
    let recorder = CostLayer::swiftide().budget(budget.clone()).layer(inner);

    metrics::with_local_recorder(&recorder, || {
        // 200 chunks embedded, then a metadata question/answer pass per chunk
        for _ in 0..200 {
            emit_usage("text-embedding-3-small", 420, 0);
            emit_usage("gpt-4o-mini", 900, 120);
        }
        // a few summaries on a bigger model
        for _ in 0..10 {
            emit_usage("gpt-4o", 6_000, 400);
        }
    });

    println!("{:<62} {:>10}", "series", "value");
    let mut rows = snapshotter.snapshot().into_vec();
    rows.sort_by_key(|row| row.0.key().to_string());
    for (key, _, _, value) in rows {
        let labels: Vec<String> = key
            .key()
            .labels()
            .filter(|l| l.key() != "pipeline")
            .map(|l| format!("{}={}", l.key(), l.value()))
            .collect();
        let name = format!("{} {{{}}}", key.key().name(), labels.join(","));
        match value {
            DebugValue::Counter(n) => println!("{name:<62} {n:>10}"),
            DebugValue::Gauge(usd) => println!("{name:<62} {:>10.6}", usd.into_inner()),
            DebugValue::Histogram(_) => {}
        }
    }

    println!();
    println!("budget: ${:.4} of ${:.2} spent, killed: {}", budget.spent_usd(), budget.limit_usd(), budget.is_killed());

    // Code that reserves before each call stops once the layer has booked the cap away
    match budget.reserve(0.01) {
        Ok(_) => println!("next call: allowed"),
        Err(e) => println!("next call: refused ({e})"),
    }
    Ok(())
}
