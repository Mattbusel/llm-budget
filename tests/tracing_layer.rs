//! CostLayer against spans shaped like the ones rig and other GenAI
//! instrumentations record.
#![allow(clippy::unwrap_used)]

use std::sync::{Arc, Mutex};

use llm_budget::tracing_layer::{CostLayer, SpanCost, COST_TARGET};
use llm_budget::{AgentBudget, ModelPrice, PriceTable, Usage};
use tracing::field::{Empty, Field, Visit};
use tracing::{Event, Subscriber};
use tracing_subscriber::layer::{Context, Layer};
use tracing_subscriber::prelude::*;
use tracing_subscriber::registry::LookupSpan;

/// Collects the fields of every `llm_budget::cost` event, with the name of
/// the span the event is in.
type Captured = Vec<(String, Vec<(String, String)>)>;

#[derive(Clone, Default)]
struct Events(Arc<Mutex<Captured>>);

struct Fields(Vec<(String, String)>);
impl Visit for Fields {
    fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
        self.0.push((field.name().to_string(), format!("{value:?}")));
    }
}

impl<S: Subscriber + for<'a> LookupSpan<'a>> Layer<S> for Events {
    fn on_event(&self, event: &Event<'_>, ctx: Context<'_, S>) {
        if event.metadata().target() != COST_TARGET {
            return;
        }
        let mut f = Fields(Vec::new());
        event.record(&mut f);
        let parent = ctx.event_span(event).map(|s| s.name().to_string()).unwrap_or_default();
        self.0.lock().unwrap().push((parent, f.0));
    }
}

impl Events {
    fn field(&self, i: usize, name: &str) -> Option<String> {
        let all = self.0.lock().unwrap();
        all.get(i)?.1.iter().find(|(k, _)| k == name).map(|(_, v)| v.clone())
    }
    fn len(&self) -> usize {
        self.0.lock().unwrap().len()
    }
}

/// A chat span the way rig-core 0.46 declares it.
macro_rules! chat_span {
    ($model:expr) => {
        tracing::info_span!(
            "chat",
            gen_ai.operation.name = "chat",
            gen_ai.provider.name = "openai",
            gen_ai.request.model = $model,
            gen_ai.response.model = Empty,
            gen_ai.usage.input_tokens = Empty,
            gen_ai.usage.output_tokens = Empty,
            gen_ai.usage.cache_read.input_tokens = Empty,
            gen_ai.usage.cache_creation.input_tokens = Empty,
            gen_ai.usage.reasoning_tokens = Empty,
        )
    };
}

fn price(model: &str, usage: Usage) -> f64 {
    PriceTable::builtin().cost_usage(model, &usage).unwrap()
}

#[test]
fn prices_a_chat_span_and_emits_the_cost_inside_it() {
    let events = Events::default();
    let layer = CostLayer::new();
    let totals = layer.tracker();
    let sub = tracing_subscriber::registry().with(layer).with(events.clone());
    tracing::subscriber::with_default(sub, || {
        let span = chat_span!("gpt-4o");
        span.record("gen_ai.response.model", "gpt-4o-2024-08-06");
        span.record("gen_ai.usage.input_tokens", 10_000u64);
        span.record("gen_ai.usage.output_tokens", 500u64);
    });
    let want = price("gpt-4o", Usage::new(10_000, 500));
    assert!((want - 0.03).abs() < 1e-9, "gpt-4o is $2.50/$10 per 1M: {want}");
    assert!((totals.total_usd() - want).abs() < 1e-9);
    assert_eq!(totals.priced_calls(), 1);
    assert_eq!(events.len(), 1);
    assert_eq!(events.0.lock().unwrap()[0].0, "chat");
    let cost: f64 = events.field(0, "gen_ai.usage.cost.amount").unwrap().parse().unwrap();
    assert!((cost - want).abs() < 1e-9);
    assert_eq!(events.field(0, "gen_ai.usage.cost.currency").as_deref(), Some("\"USD\""));
    assert_eq!(events.field(0, "gen_ai.usage.cost.source").as_deref(), Some("\"local\""));
    // the pre-0.7 name is no longer written
    assert!(events.field(0, "gen_ai.usage.cost").is_none());
    assert_eq!(events.field(0, "gen_ai.provider.name").as_deref(), Some("\"openai\""));
}

#[test]
fn cache_reads_get_the_cache_price() {
    let layer = CostLayer::new();
    let totals = layer.tracker();
    let sub = tracing_subscriber::registry().with(layer);
    tracing::subscriber::with_default(sub, || {
        let span = chat_span!("claude-sonnet-4-5");
        span.record("gen_ai.usage.input_tokens", 10_000u64);
        span.record("gen_ai.usage.cache_read.input_tokens", 8_000u64);
        span.record("gen_ai.usage.output_tokens", 200u64);
    });
    let cached = price("claude-sonnet-4-5", Usage::from_parts(2_000, 8_000, 0, 200));
    let uncached = price("claude-sonnet-4-5", Usage::new(10_000, 200));
    assert!(cached < uncached);
    assert!((totals.total_usd() - cached).abs() < 1e-9);
}

#[test]
fn agent_span_sum_is_not_counted_twice() {
    let layer = CostLayer::new();
    let totals = layer.tracker();
    let sub = tracing_subscriber::registry().with(layer);
    tracing::subscriber::with_default(sub, || {
        // rig's invoke_agent span: no model, the run's summed usage.
        let agent = tracing::info_span!(
            "invoke_agent",
            gen_ai.operation.name = "invoke_agent",
            gen_ai.request.model = "gpt-4o",
            gen_ai.usage.input_tokens = Empty,
            gen_ai.usage.output_tokens = Empty,
        );
        agent.in_scope(|| {
            for _ in 0..2 {
                let turn = chat_span!("gpt-4o");
                turn.record("gen_ai.usage.input_tokens", 1_000u64);
                turn.record("gen_ai.usage.output_tokens", 100u64);
            }
        });
        agent.record("gen_ai.usage.input_tokens", 2_000u64);
        agent.record("gen_ai.usage.output_tokens", 200u64);
    });
    assert_eq!(totals.priced_calls(), 2);
    let want = 2.0 * price("gpt-4o", Usage::new(1_000, 100));
    assert!((totals.total_usd() - want).abs() < 1e-9);
}

#[test]
fn unknown_model_is_counted_not_guessed() {
    let layer = CostLayer::new();
    let totals = layer.tracker();
    let sub = tracing_subscriber::registry().with(layer);
    tracing::subscriber::with_default(sub, || {
        let span = chat_span!("my-finetune-v3");
        span.record("gen_ai.usage.input_tokens", 1_000u64);
        span.record("gen_ai.usage.output_tokens", 10u64);
    });
    assert_eq!(totals.priced_calls(), 0);
    assert_eq!(totals.unpriced_calls(), 1);
    assert_eq!(totals.total_usd(), 0.0);
}

#[test]
fn custom_table_prices_your_own_models() {
    let table = PriceTable::new().with_price(ModelPrice::new("my-finetune-v3", "self-hosted", 1.0, 2.0));
    let layer = CostLayer::new().price_table(table);
    let totals = layer.tracker();
    let sub = tracing_subscriber::registry().with(layer);
    tracing::subscriber::with_default(sub, || {
        let span = chat_span!("my-finetune-v3");
        span.record("gen_ai.usage.input_tokens", 1_000_000u64);
        span.record("gen_ai.usage.output_tokens", 500_000u64);
    });
    assert!((totals.total_usd() - 2.0).abs() < 1e-9);
}

#[test]
fn spans_without_usage_are_ignored() {
    let events = Events::default();
    let layer = CostLayer::new();
    let totals = layer.tracker();
    let sub = tracing_subscriber::registry().with(layer).with(events.clone());
    tracing::subscriber::with_default(sub, || {
        let _plain = tracing::info_span!("db_query", rows = 3).entered();
        // declared but never recorded (the call failed)
        let _failed = chat_span!("gpt-4o");
    });
    assert_eq!(totals.priced_calls() + totals.unpriced_calls(), 0);
    assert_eq!(events.len(), 0);
}

#[test]
fn a_reported_cost_on_the_span_wins() {
    let layer = CostLayer::new();
    let totals = layer.tracker();
    let sub = tracing_subscriber::registry().with(layer);
    tracing::subscriber::with_default(sub, || {
        let span = tracing::info_span!(
            "chat",
            gen_ai.request.model = "gpt-4o",
            gen_ai.usage.input_tokens = 10_000u64,
            gen_ai.usage.output_tokens = 500u64,
            gen_ai.usage.cost = Empty,
        );
        span.record("gen_ai.usage.cost", 0.0123f64);
    });
    assert!((totals.total_usd() - 0.0123).abs() < 1e-9);
}

#[test]
fn a_reported_semconv_amount_and_source_are_kept() {
    let events = Events::default();
    let layer = CostLayer::new();
    let totals = layer.tracker();
    let seen: Arc<Mutex<Vec<SpanCost>>> = Arc::default();
    let sink = seen.clone();
    let layer = layer.on_cost(move |c| sink.lock().unwrap().push(c.clone()));
    let sub = tracing_subscriber::registry().with(layer).with(events.clone());
    tracing::subscriber::with_default(sub, || {
        let span = tracing::info_span!(
            "chat",
            gen_ai.request.model = "gpt-4o",
            gen_ai.usage.input_tokens = 10_000u64,
            gen_ai.usage.output_tokens = 500u64,
            gen_ai.usage.cost.amount = Empty,
            gen_ai.usage.cost.currency = "USD",
            gen_ai.usage.cost.source = "provider",
        );
        span.record("gen_ai.usage.cost.amount", 0.0123f64);
    });
    assert!((totals.total_usd() - 0.0123).abs() < 1e-9);
    let c = seen.lock().unwrap()[0].clone();
    assert!(c.reported);
    assert_eq!(c.source, Some(llm_budget::semconv::CostSource::Provider));
    assert_eq!(events.field(0, "gen_ai.usage.cost.source").as_deref(), Some("\"provider\""));
}

#[test]
fn a_reported_amount_in_another_currency_is_repriced_in_usd() {
    let layer = CostLayer::new();
    let totals = layer.tracker();
    let sub = tracing_subscriber::registry().with(layer);
    tracing::subscriber::with_default(sub, || {
        let span = tracing::info_span!(
            "chat",
            gen_ai.request.model = "gpt-4o",
            gen_ai.usage.input_tokens = 10_000u64,
            gen_ai.usage.output_tokens = 500u64,
            gen_ai.usage.cost.amount = Empty,
            gen_ai.usage.cost.currency = "EUR",
        );
        span.record("gen_ai.usage.cost.amount", 9.0f64);
    });
    assert!((totals.total_usd() - price("gpt-4o", Usage::new(10_000, 500))).abs() < 1e-9);
}

#[test]
fn older_prompt_and_completion_token_names() {
    let layer = CostLayer::new();
    let totals = layer.tracker();
    let sub = tracing_subscriber::registry().with(layer);
    tracing::subscriber::with_default(sub, || {
        let _span = tracing::info_span!(
            "openai.chat",
            gen_ai.system = "openai",
            gen_ai.request.model = "gpt-4o",
            gen_ai.usage.prompt_tokens = 10_000i64,
            gen_ai.usage.completion_tokens = 500i64,
        );
    });
    assert!((totals.total_usd() - price("gpt-4o", Usage::new(10_000, 500))).abs() < 1e-9);
}

#[test]
fn budget_is_booked_and_killed_at_the_cap() {
    let budget = Arc::new(AgentBudget::builder("traced", 0.05).build().unwrap());
    let seen: Arc<Mutex<Vec<SpanCost>>> = Arc::default();
    let seen2 = seen.clone();
    let events = Events::default();
    let layer = CostLayer::new().budget(budget.clone()).on_cost(move |c| seen2.lock().unwrap().push(c.clone()));
    let totals = layer.tracker();
    let sub = tracing_subscriber::registry().with(layer).with(events.clone());
    tracing::subscriber::with_default(sub, || {
        for _ in 0..2 {
            // $0.03 each on gpt-4o
            let span = chat_span!("gpt-4o");
            span.record("gen_ai.usage.input_tokens", 10_000u64);
            span.record("gen_ai.usage.output_tokens", 500u64);
        }
    });
    // Both calls were paid for, so both are booked, even past the cap.
    assert!((budget.spent_usd() - 0.06).abs() < 1e-9);
    assert!(budget.is_killed());
    assert!(totals.over_cap());
    // The next call on this budget is refused before it is made.
    assert!(budget.reserve_for("gpt-4o", &Usage::new(100, 10)).is_err());

    let seen = seen.lock().unwrap();
    assert_eq!(seen.len(), 2);
    assert!(!seen[0].outcome.as_ref().unwrap().over_cap);
    assert!(seen[1].outcome.as_ref().unwrap().over_cap);
    assert_eq!(seen[1].model, "gpt-4o");
    // two cost events and one cap warning
    assert_eq!(events.len(), 3);
}

#[test]
fn kill_on_cap_false_books_without_killing() {
    let budget = Arc::new(AgentBudget::builder("traced", 0.01).build().unwrap());
    let layer = CostLayer::new().budget(budget.clone()).kill_on_cap(false).emit_events(false);
    let totals = layer.tracker();
    let sub = tracing_subscriber::registry().with(layer);
    tracing::subscriber::with_default(sub, || {
        let span = chat_span!("gpt-4o");
        span.record("gen_ai.usage.input_tokens", 10_000u64);
        span.record("gen_ai.usage.output_tokens", 500u64);
    });
    assert!(totals.over_cap());
    assert!(!budget.is_killed());
    assert!((budget.spent_usd() - 0.03).abs() < 1e-9);
}
