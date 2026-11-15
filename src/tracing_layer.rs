//! Price every LLM call your app already traces (feature `tracing-layer`).
//!
//! [`CostLayer`] is a [`tracing_subscriber::Layer`]. It watches spans that
//! carry the OpenTelemetry GenAI usage fields (`gen_ai.usage.input_tokens`,
//! `gen_ai.usage.output_tokens`, ...), which rig, and any client that follows
//! the GenAI semantic conventions, already records on every model call. When
//! such a span closes, the layer prices the call with llm-prices (through a
//! [`PriceTable`]) and:
//!
//! - emits an `INFO` event on target `llm_budget::cost`, inside the span, with
//!   `gen_ai.usage.cost` (USD) next to the model and token counts, so the cost
//!   lands wherever your spans go (fmt logs, OpenTelemetry, Langfuse, ...);
//! - adds it to the running totals of a [`CostTracker`];
//! - optionally books it on an [`AgentBudget`] and, once spend is over the
//!   cap, trips the budget's kill switch so every `reserve_for`, `charge_*`
//!   or `rig::BudgetHook` call on that budget refuses the next model call.
//!
//! Nothing in the instrumented code changes: no hook, wrapper or client type.
//!
//! ```
//! use std::sync::Arc;
//! use llm_budget::tracing_layer::CostLayer;
//! use llm_budget::AgentBudget;
//! use tracing_subscriber::prelude::*;
//!
//! # fn main() -> Result<(), llm_budget::BudgetError> {
//! let budget = Arc::new(AgentBudget::builder("my-app", 5.00).build()?);
//! let cost = CostLayer::new().budget(budget.clone());
//! let totals = cost.tracker();
//!
//! // Add CostLayer before an OpenTelemetry layer so the cost event is
//! // exported with its span.
//! let subscriber = tracing_subscriber::registry().with(cost);
//! tracing::subscriber::with_default(subscriber, || {
//!     // What rig records on a chat span:
//!     let span = tracing::info_span!(
//!         "chat gpt-4o",
//!         gen_ai.operation.name = "chat",
//!         gen_ai.request.model = "gpt-4o",
//!         gen_ai.usage.input_tokens = tracing::field::Empty,
//!         gen_ai.usage.output_tokens = tracing::field::Empty,
//!     );
//!     span.record("gen_ai.usage.input_tokens", 10_000u64);
//!     span.record("gen_ai.usage.output_tokens", 500u64);
//! }); // the span closes here and is priced
//!
//! assert!((totals.total_usd() - 0.03).abs() < 1e-9); // $2.50/M in, $10/M out
//! assert!((budget.spent_usd() - 0.03).abs() < 1e-9);
//! # Ok(())
//! # }
//! ```
//!
//! # Which spans are priced
//!
//! A span is watched when it declares `gen_ai.usage.input_tokens` or
//! `gen_ai.usage.output_tokens` (or the older `gen_ai.usage.prompt_tokens` /
//! `gen_ai.usage.completion_tokens`). It is priced when it closes, if a token
//! count was recorded and a model is known:
//!
//! - **Model**: `gen_ai.response.model` (the exact id the provider answered
//!   with), then `gen_ai.request.model` if the response id has no price.
//! - **Tokens**: `gen_ai.usage.input_tokens` counts every prompt token, with
//!   `gen_ai.usage.cache_read.input_tokens` and
//!   `gen_ai.usage.cache_creation.input_tokens` as parts of it (the GenAI
//!   conventions and rig both count this way), so cache reads get the cache
//!   price. Reasoning tokens are part of `gen_ai.usage.output_tokens`.
//! - **Provider figure first**: if the span itself declares and records
//!   `gen_ai.usage.cost`, that number is used instead of a price lookup.
//! - **No double counting**: a span whose descendants were already priced is
//!   skipped. rig records each turn on a chat span and the run's sum on the
//!   `invoke_agent` span around them; only the turns are counted.
//!
//! A span with usage but a model no source can price is counted in
//! [`CostTracker::unpriced_calls`] and reported in a `DEBUG` event, never
//! guessed. Give such models a price with [`PriceTable::with_price`] or
//! [`PriceTable::with_hook`] and pass the table to
//! [`CostLayer::price_table`].
//!
//! # A cap, not a gate
//!
//! A layer sees a call only after it returned, so it cannot refuse it. With
//! [`CostLayer::budget`], every priced call is booked (even past the cap,
//! since it was paid for). When spend goes over the cap the layer kills the
//! budget (unless [`CostLayer::kill_on_cap`] is `false`) and emits a `WARN`
//! event. Code that checks the same budget before each call, such as
//! [`AgentBudget::reserve_for`] or `rig::BudgetHook`, then stops. On its own,
//! check [`CostTracker::over_cap`] or [`AgentBudget::is_killed`] between
//! calls.

use std::fmt;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

use tracing::field::{Field, Visit};
use tracing::span::{Attributes, Id, Record};
use tracing::Subscriber;
use tracing_subscriber::layer::Context;
use tracing_subscriber::registry::LookupSpan;
use tracing_subscriber::Layer;

use crate::budget::{AgentBudget, ChargeContext, ChargeOutcome};
use crate::pricing::PriceTable;
use crate::usage::Usage;

/// Target of the events [`CostLayer`] emits.
pub const COST_TARGET: &str = "llm_budget::cost";

const INPUT: &[&str] = &["gen_ai.usage.input_tokens", "gen_ai.usage.prompt_tokens"];
const OUTPUT: &[&str] = &["gen_ai.usage.output_tokens", "gen_ai.usage.completion_tokens"];
const CACHE_READ: &str = "gen_ai.usage.cache_read.input_tokens";
const CACHE_WRITE: &str = "gen_ai.usage.cache_creation.input_tokens";
const RESPONSE_MODEL: &str = "gen_ai.response.model";
const REQUEST_MODEL: &str = "gen_ai.request.model";
const PROVIDER: &[&str] = &["gen_ai.provider.name", "gen_ai.system"];
const COST: &str = "gen_ai.usage.cost";

/// What one priced span cost, handed to [`CostLayer::on_cost`].
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct SpanCost {
    /// The span's name (rig: `chat <model>`).
    pub span_name: &'static str,
    /// The model the call was priced as.
    pub model: String,
    /// `gen_ai.provider.name` (or `gen_ai.system`), when recorded.
    pub provider: Option<String>,
    /// The token counts read from the span.
    pub usage: Usage,
    /// Cost of the call in USD.
    pub cost_usd: f64,
    /// True when the cost came from the span's own `gen_ai.usage.cost`.
    pub reported: bool,
    /// The booking on the budget, when [`CostLayer::budget`] is set.
    pub outcome: Option<ChargeOutcome>,
}

type OnCost = dyn Fn(&SpanCost) + Send + Sync;

/// Running totals of a [`CostLayer`], readable from anywhere while the layer
/// is installed. Get one with [`CostLayer::tracker`]; clones share the totals.
#[derive(Clone, Default)]
pub struct CostTracker(Arc<Totals>);

#[derive(Default)]
struct Totals {
    nano_usd: AtomicU64,
    calls: AtomicU64,
    unpriced: AtomicU64,
    over_cap: AtomicBool,
}

impl CostTracker {
    /// Total priced spend seen by the layer, in USD.
    pub fn total_usd(&self) -> f64 {
        self.0.nano_usd.load(Ordering::Relaxed) as f64 / 1e9
    }

    /// Number of calls priced.
    pub fn priced_calls(&self) -> u64 {
        self.0.calls.load(Ordering::Relaxed)
    }

    /// Calls that reported usage for a model no source could price.
    pub fn unpriced_calls(&self) -> u64 {
        self.0.unpriced.load(Ordering::Relaxed)
    }

    /// True once a booking on the layer's budget went over its cap.
    pub fn over_cap(&self) -> bool {
        self.0.over_cap.load(Ordering::Relaxed)
    }

    fn add(&self, usd: f64) {
        // whole nano-dollars, like the budget's own books
        let nanos = (usd * 1e9).round().max(0.0) as u64;
        self.0.nano_usd.fetch_add(nanos, Ordering::Relaxed);
        self.0.calls.fetch_add(1, Ordering::Relaxed);
    }
}

impl fmt::Debug for CostTracker {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CostTracker")
            .field("total_usd", &self.total_usd())
            .field("priced_calls", &self.priced_calls())
            .field("unpriced_calls", &self.unpriced_calls())
            .field("over_cap", &self.over_cap())
            .finish()
    }
}

/// A [`Layer`] that prices spans carrying OpenTelemetry GenAI usage fields.
/// See the [module docs](self).
#[derive(Clone)]
pub struct CostLayer {
    table: Option<Arc<PriceTable>>,
    budget: Option<Arc<AgentBudget>>,
    kill_on_cap: bool,
    emit_events: bool,
    on_cost: Option<Arc<OnCost>>,
    tracker: CostTracker,
}

impl Default for CostLayer {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for CostLayer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CostLayer")
            .field("custom_table", &self.table.is_some())
            .field("budget", &self.budget.as_ref().map(|b| b.limit_usd()))
            .field("kill_on_cap", &self.kill_on_cap)
            .field("emit_events", &self.emit_events)
            .field("tracker", &self.tracker)
            .finish()
    }
}

impl CostLayer {
    /// A layer pricing from the built-in table (llm-prices first), with no
    /// budget, emitting a cost event per priced call.
    pub fn new() -> Self {
        Self {
            table: None,
            budget: None,
            kill_on_cap: true,
            emit_events: true,
            on_cost: None,
            tracker: CostTracker::default(),
        }
    }

    /// Price with `table` instead of the built-in one (your own prices, a
    /// hook for fine-tuned or self-hosted models, ...).
    pub fn price_table(mut self, table: impl Into<Arc<PriceTable>>) -> Self {
        self.table = Some(table.into());
        self
    }

    /// Book every priced call on `budget`, and trip its kill switch once
    /// spend is over the cap (see [`kill_on_cap`](Self::kill_on_cap)).
    pub fn budget(mut self, budget: Arc<AgentBudget>) -> Self {
        self.budget = Some(budget);
        self
    }

    /// Whether going over the budget's cap kills the budget (default
    /// `true`). With `false` the spend is still booked and
    /// [`CostTracker::over_cap`] still turns on.
    pub fn kill_on_cap(mut self, on: bool) -> Self {
        self.kill_on_cap = on;
        self
    }

    /// Whether to emit the `llm_budget::cost` event for each priced call
    /// (default `true`).
    pub fn emit_events(mut self, on: bool) -> Self {
        self.emit_events = on;
        self
    }

    /// Call `f` for every priced span (after it is booked).
    pub fn on_cost<F>(mut self, f: F) -> Self
    where
        F: Fn(&SpanCost) + Send + Sync + 'static,
    {
        self.on_cost = Some(Arc::new(f));
        self
    }

    /// A handle on this layer's running totals. Take it before installing
    /// the layer.
    pub fn tracker(&self) -> CostTracker {
        self.tracker.clone()
    }

    fn table(&self) -> &PriceTable {
        self.table.as_deref().unwrap_or_else(|| PriceTable::builtin())
    }

    /// Price `data`: the span's own cost if it reported one, else the first
    /// model id with a price.
    fn price(&self, data: &GenAiSpan, usage: &Usage) -> Option<(String, f64, bool)> {
        let model = data.response_model.as_deref().or(data.request_model.as_deref());
        if let Some(cost) = data.reported_cost.filter(|c| c.is_finite() && *c >= 0.0) {
            return Some((model.unwrap_or("unknown").to_string(), cost, true));
        }
        let table = self.table();
        [data.response_model.as_deref(), data.request_model.as_deref()]
            .into_iter()
            .flatten()
            .find_map(|m| table.cost_usage(m, usage).ok().map(|usd| (m.to_string(), usd, false)))
    }
}

/// What the layer has seen of one watched span.
#[derive(Default)]
struct GenAiSpan {
    input: Option<u64>,
    output: Option<u64>,
    cache_read: Option<u64>,
    cache_write: Option<u64>,
    response_model: Option<String>,
    request_model: Option<String>,
    provider: Option<String>,
    reported_cost: Option<f64>,
    /// A descendant was already priced; this span's usage is an aggregate.
    child_priced: bool,
}

impl GenAiSpan {
    fn usage(&self) -> Option<Usage> {
        if self.input.is_none() && self.output.is_none() {
            return None;
        }
        let input = self.input.unwrap_or(0);
        let read = self.cache_read.unwrap_or(0).min(input);
        let write = self.cache_write.unwrap_or(0).min(input - read);
        Some(Usage {
            input_tokens: input,
            cached_input_tokens: read,
            cache_write_tokens: write,
            output_tokens: self.output.unwrap_or(0),
        })
    }

    fn set_u64(&mut self, name: &str, v: u64) {
        if INPUT.contains(&name) {
            self.input = Some(v);
        } else if OUTPUT.contains(&name) {
            self.output = Some(v);
        } else if name == CACHE_READ {
            self.cache_read = Some(v);
        } else if name == CACHE_WRITE {
            self.cache_write = Some(v);
        } else if name == COST {
            self.reported_cost = Some(v as f64);
        }
    }

    fn set_str(&mut self, name: &str, v: &str) {
        if name == RESPONSE_MODEL {
            self.response_model = Some(v.to_string());
        } else if name == REQUEST_MODEL {
            self.request_model = Some(v.to_string());
        } else if PROVIDER.contains(&name) {
            self.provider = Some(v.to_string());
        } else if let Ok(n) = v.trim().parse::<u64>() {
            self.set_u64(name, n);
        } else if name == COST {
            self.reported_cost = v.trim().parse::<f64>().ok();
        }
    }
}

impl Visit for GenAiSpan {
    fn record_u64(&mut self, field: &Field, value: u64) {
        self.set_u64(field.name(), value);
    }

    fn record_i64(&mut self, field: &Field, value: i64) {
        if let Ok(v) = u64::try_from(value) {
            self.set_u64(field.name(), v);
        }
    }

    fn record_f64(&mut self, field: &Field, value: f64) {
        if field.name() == COST {
            self.reported_cost = Some(value);
        }
    }

    fn record_str(&mut self, field: &Field, value: &str) {
        self.set_str(field.name(), value);
    }

    fn record_debug(&mut self, field: &Field, value: &dyn fmt::Debug) {
        let text = format!("{value:?}");
        self.set_str(field.name(), text.trim_matches('"'));
    }
}

fn is_watched(fields: &tracing::field::FieldSet) -> bool {
    INPUT.iter().chain(OUTPUT).any(|name| fields.field(name).is_some())
}

impl<S> Layer<S> for CostLayer
where
    S: Subscriber + for<'a> LookupSpan<'a>,
{
    fn on_new_span(&self, attrs: &Attributes<'_>, id: &Id, ctx: Context<'_, S>) {
        if !is_watched(attrs.metadata().fields()) {
            return;
        }
        let Some(span) = ctx.span(id) else { return };
        let mut data = GenAiSpan::default();
        attrs.record(&mut data);
        span.extensions_mut().insert(data);
    }

    fn on_record(&self, id: &Id, values: &Record<'_>, ctx: Context<'_, S>) {
        let Some(span) = ctx.span(id) else { return };
        let mut ext = span.extensions_mut();
        if let Some(data) = ext.get_mut::<GenAiSpan>() {
            values.record(data);
        }
    }

    fn on_close(&self, id: Id, ctx: Context<'_, S>) {
        let Some(span) = ctx.span(&id) else { return };
        let Some(data) = span.extensions_mut().remove::<GenAiSpan>() else { return };
        if data.child_priced {
            return;
        }
        let Some(usage) = data.usage() else { return };
        let Some((model, cost_usd, reported)) = self.price(&data, &usage) else {
            let model = data.response_model.as_deref().or(data.request_model.as_deref());
            if let Some(model) = model {
                self.tracker.0.unpriced.fetch_add(1, Ordering::Relaxed);
                if self.emit_events {
                    tracing::event!(
                        target: COST_TARGET,
                        parent: &id,
                        tracing::Level::DEBUG,
                        gen_ai.request.model = model,
                        gen_ai.usage.input_tokens = usage.input_tokens,
                        gen_ai.usage.output_tokens = usage.output_tokens,
                        "no price for this model; call not counted"
                    );
                }
            }
            return;
        };

        // Ancestors that also carry usage hold a sum that includes this call.
        for parent in span.scope().skip(1) {
            if let Some(p) = parent.extensions_mut().get_mut::<GenAiSpan>() {
                p.child_priced = true;
            }
        }

        self.tracker.add(cost_usd);
        let outcome = self.budget.as_ref().map(|b| {
            let out = b.charge_usd_force(cost_usd, ChargeContext::new().model(model.as_str()).usage(&usage));
            if out.over_cap {
                self.tracker.0.over_cap.store(true, Ordering::Relaxed);
                if self.kill_on_cap && !b.is_killed() {
                    b.kill("LLM spend traced by CostLayer is over the budget cap");
                    if self.emit_events {
                        tracing::event!(
                            target: COST_TARGET,
                            parent: &id,
                            tracing::Level::WARN,
                            spent_usd = out.total_spent_usd,
                            limit_usd = b.limit_usd(),
                            "LLM budget cap reached; budget killed"
                        );
                    }
                }
            }
            out
        });

        if self.emit_events {
            tracing::event!(
                target: COST_TARGET,
                parent: &id,
                tracing::Level::INFO,
                gen_ai.usage.cost = cost_usd,
                gen_ai.request.model = model.as_str(),
                gen_ai.provider.name = data.provider.as_deref(),
                gen_ai.usage.input_tokens = usage.input_tokens,
                gen_ai.usage.output_tokens = usage.output_tokens,
                gen_ai.usage.cache_read.input_tokens = usage.cached_input_tokens,
                gen_ai.usage.cache_creation.input_tokens = usage.cache_write_tokens,
                llm_budget.total_usd = self.tracker.total_usd(),
                llm_budget.over_cap = outcome.as_ref().is_some_and(|o| o.over_cap),
                "llm call priced"
            );
        }

        if let Some(f) = &self.on_cost {
            f(&SpanCost { span_name: span.name(), model, provider: data.provider, usage, cost_usd, reported, outcome });
        }
    }
}
