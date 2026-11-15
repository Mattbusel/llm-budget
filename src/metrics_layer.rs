//! Turn token counters into dollar cost, inside your metrics pipeline.
//!
//! Many frameworks already count tokens through the
//! [metrics](https://crates.io/crates/metrics) facade, labelled with the model
//! that used them. [`CostLayer`] is a [`metrics_util::layers::Layer`]: wrap
//! your recorder (Prometheus, StatsD, OpenTelemetry, a test recorder, ...)
//! with it, and every increment of a token counter is also priced and added
//! to a cost metric with the same labels. Nothing in the framework changes.
//!
//! It is the metrics twin of `tracing_layer::CostLayer` (feature
//! `tracing-layer`), for code that reports usage as counters rather than span
//! fields.
//!
//! For [swiftide](https://crates.io/crates/swiftide) (feature `metrics`),
//! which counts `swiftide.usage.prompt_tokens` and
//! `swiftide.usage.completion_tokens` with a `model` label,
//! [`CostLayer::swiftide`] is preset:
//!
//! ```
//! use std::sync::Arc;
//! use llm_budget::metrics_layer::CostLayer;
//! use llm_budget::AgentBudget;
//! use metrics_util::debugging::DebuggingRecorder;
//! use metrics_util::layers::Layer as _;
//!
//! # fn main() -> Result<(), llm_budget::BudgetError> {
//! let budget = Arc::new(AgentBudget::builder("indexing", 5.00).build()?);
//! let recorder = CostLayer::swiftide()
//!     .budget(budget.clone())
//!     .layer(DebuggingRecorder::new()); // your exporter's recorder here
//!
//! metrics::with_local_recorder(&recorder, || {
//!     // what swiftide emits after a call to gpt-4o: 10k prompt, 1k completion tokens
//!     metrics::counter!("swiftide.usage.prompt_tokens", "model" => "gpt-4o").increment(10_000);
//!     metrics::counter!("swiftide.usage.completion_tokens", "model" => "gpt-4o").increment(1_000);
//! });
//!
//! // $2.50/M input and $10/M output: $0.025 + $0.01
//! assert!((budget.spent_usd() - 0.035).abs() < 1e-9);
//! # Ok(())
//! # }
//! ```
//!
//! Any other emitter works the same way: name its counters with
//! [`input_tokens`](CostLayer::input_tokens) and
//! [`output_tokens`](CostLayer::output_tokens), and its model label with
//! [`model_label`](CostLayer::model_label).
//!
//! # What it records
//!
//! | Name | Type | Labels | Meaning |
//! |---|---|---|---|
//! | [`cost_metric`](CostLayer::cost_metric) (`llm.usage.cost_usd`, or `swiftide.usage.cost_usd` for the preset) | gauge, only goes up | the token counter's labels | USD spent |
//! | `llm_budget_unpriced_tokens_total` | counter | the token counter's labels | tokens on a model with no price |
//!
//! The cost is a gauge because metrics counters only hold whole numbers; it is
//! incremented, never set, so a rate over it is spend per second. Prices come
//! from the [`PriceTable`] (the built-in one by default: llm-prices, then the
//! LiteLLM snapshot), and with [`budget`](CostLayer::budget) every priced
//! increment is also booked on an [`AgentBudget`].
//!
//! # Limits
//!
//! - Input and output tokens are priced as they arrive, each on its own, so a
//!   long-context tier is chosen from the prompt increment alone and output
//!   tokens are always billed at the base output price. Prompt-cache
//!   discounts are not applied (the counters do not say which tokens were
//!   cached), so cached prompts are priced at the full input rate.
//! - Only `increment` is priced. A counter set with `absolute` is passed on
//!   unpriced.
//! - Metrics are emitted after the call has been paid for, so the budget can
//!   only stop the *next* call: once it is over its cap the layer trips the
//!   kill switch, and code that reserves before each call
//!   ([`AgentBudget::reserve_for`]) or checks [`AgentBudget::is_killed`] stops
//!   there.

use std::sync::Arc;

use metrics::{Counter, CounterFn, Gauge, Histogram, Key, KeyName, Label, Metadata, Recorder, SharedString, Unit};
use metrics_util::layers::Layer;

use crate::budget::{AgentBudget, ChargeContext};
use crate::pricing::PriceTable;
use crate::usage::Usage;

/// Name of the counter for tokens whose model has no price.
pub const UNPRICED_TOKENS: &str = "llm_budget_unpriced_tokens_total";

/// A [`Layer`] that prices token counters into a cost metric. See the
/// [module docs](self).
#[derive(Clone, Debug)]
pub struct CostLayer {
    config: Arc<Config>,
}

struct Config {
    input_counters: Vec<String>,
    output_counters: Vec<String>,
    model_label: String,
    cost_metric: String,
    prices: Option<Arc<PriceTable>>,
    budget: Option<Arc<AgentBudget>>,
    kill_on_over_cap: bool,
}

impl std::fmt::Debug for Config {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CostLayer")
            .field("input_counters", &self.input_counters)
            .field("output_counters", &self.output_counters)
            .field("model_label", &self.model_label)
            .field("cost_metric", &self.cost_metric)
            .field("budget", &self.budget.as_ref().map(|b| b.limit_usd()))
            .field("kill_on_over_cap", &self.kill_on_over_cap)
            .finish_non_exhaustive()
    }
}

impl Default for CostLayer {
    fn default() -> Self {
        Self::new()
    }
}

impl CostLayer {
    /// A layer with no token counters yet, the model read from the `model`
    /// label and cost recorded as `llm.usage.cost_usd`.
    pub fn new() -> Self {
        Self {
            config: Arc::new(Config {
                input_counters: Vec::new(),
                output_counters: Vec::new(),
                model_label: "model".to_string(),
                cost_metric: "llm.usage.cost_usd".to_string(),
                prices: None,
                budget: None,
                kill_on_over_cap: true,
            }),
        }
    }

    /// Preset for swiftide's usage metrics (`swiftide.usage.prompt_tokens`
    /// and `swiftide.usage.completion_tokens`, labelled `model`); cost is
    /// recorded as `swiftide.usage.cost_usd`.
    pub fn swiftide() -> Self {
        Self::new()
            .input_tokens("swiftide.usage.prompt_tokens")
            .output_tokens("swiftide.usage.completion_tokens")
            .cost_metric("swiftide.usage.cost_usd")
    }

    fn config_mut(&mut self) -> &mut Config {
        Arc::make_mut(&mut self.config)
    }

    /// Price increments of the counter `name` as input (prompt) tokens. Can be
    /// called more than once.
    #[must_use]
    pub fn input_tokens(mut self, name: impl Into<String>) -> Self {
        self.config_mut().input_counters.push(name.into());
        self
    }

    /// Price increments of the counter `name` as output (completion) tokens.
    /// Can be called more than once.
    #[must_use]
    pub fn output_tokens(mut self, name: impl Into<String>) -> Self {
        self.config_mut().output_counters.push(name.into());
        self
    }

    /// The label that names the model (default `model`). Increments without
    /// it are passed on unpriced.
    #[must_use]
    pub fn model_label(mut self, label: impl Into<String>) -> Self {
        self.config_mut().model_label = label.into();
        self
    }

    /// Name of the cost gauge (default `llm.usage.cost_usd`).
    #[must_use]
    pub fn cost_metric(mut self, name: impl Into<String>) -> Self {
        self.config_mut().cost_metric = name.into();
        self
    }

    /// Price with this table instead of the built-in one, for your own or
    /// negotiated prices.
    #[must_use]
    pub fn price_table(mut self, table: impl Into<Arc<PriceTable>>) -> Self {
        self.config_mut().prices = Some(table.into());
        self
    }

    /// Also book every priced increment on `budget` (as spend that already
    /// happened, so it is booked even past the cap). Once the budget is over
    /// its cap the layer trips its kill switch, unless
    /// [`kill_on_cap(false)`](Self::kill_on_cap) is set.
    #[must_use]
    pub fn budget(mut self, budget: Arc<AgentBudget>) -> Self {
        self.config_mut().budget = Some(budget);
        self
    }

    /// Whether to trip the budget's kill switch once spend is over its cap
    /// (default `true`). With `false` the layer only books.
    #[must_use]
    pub fn kill_on_cap(mut self, on: bool) -> Self {
        self.config_mut().kill_on_over_cap = on;
        self
    }
}

impl Clone for Config {
    fn clone(&self) -> Self {
        Self {
            input_counters: self.input_counters.clone(),
            output_counters: self.output_counters.clone(),
            model_label: self.model_label.clone(),
            cost_metric: self.cost_metric.clone(),
            prices: self.prices.clone(),
            budget: self.budget.clone(),
            kill_on_over_cap: self.kill_on_over_cap,
        }
    }
}

impl<R> Layer<R> for CostLayer {
    type Output = CostRecorder<R>;

    fn layer(&self, inner: R) -> Self::Output {
        CostRecorder { inner, config: self.config.clone() }
    }
}

/// The recorder [`CostLayer`] produces: forwards everything to the wrapped
/// recorder and adds the cost metric.
#[derive(Debug)]
pub struct CostRecorder<R> {
    inner: R,
    config: Arc<Config>,
}

impl<R> CostRecorder<R> {
    /// The wrapped recorder.
    pub fn inner(&self) -> &R {
        &self.inner
    }

    /// Unwrap into the wrapped recorder.
    pub fn into_inner(self) -> R {
        self.inner
    }
}

#[derive(Clone, Copy, Debug)]
enum Side {
    Input,
    Output,
}

impl Config {
    fn side(&self, name: &str) -> Option<Side> {
        if self.input_counters.iter().any(|c| c == name) {
            Some(Side::Input)
        } else if self.output_counters.iter().any(|c| c == name) {
            Some(Side::Output)
        } else {
            None
        }
    }

    fn prices(&self) -> &PriceTable {
        match &self.prices {
            Some(table) => table,
            None => PriceTable::builtin(),
        }
    }
}

impl<R: Recorder> Recorder for CostRecorder<R> {
    fn describe_counter(&self, key: KeyName, unit: Option<Unit>, description: SharedString) {
        if self.config.side(key.as_str()).is_some() {
            self.inner.describe_gauge(
                KeyName::from(self.config.cost_metric.clone()),
                None,
                SharedString::from("LLM spend in USD, priced from token counts by llm-budget"),
            );
        }
        self.inner.describe_counter(key, unit, description);
    }

    fn describe_gauge(&self, key: KeyName, unit: Option<Unit>, description: SharedString) {
        self.inner.describe_gauge(key, unit, description);
    }

    fn describe_histogram(&self, key: KeyName, unit: Option<Unit>, description: SharedString) {
        self.inner.describe_histogram(key, unit, description);
    }

    fn register_counter(&self, key: &Key, metadata: &Metadata<'_>) -> Counter {
        let counter = self.inner.register_counter(key, metadata);
        let Some(side) = self.config.side(key.name()) else {
            return counter;
        };
        let Some(model) = key.labels().find(|l| l.key() == self.config.model_label).map(|l| l.value().to_string())
        else {
            return counter;
        };
        let labels: Vec<Label> = key.labels().cloned().collect();
        // Only one of the two series exists per model, so unpriced models do not show a $0 cost
        // and priced ones do not carry an empty unpriced counter.
        let target = if self.config.prices().cost_usage(&model, &Usage::new(1, 0)).is_ok() {
            Target::Cost(self.inner.register_gauge(&Key::from_parts(self.config.cost_metric.clone(), labels), metadata))
        } else {
            Target::Unpriced(self.inner.register_counter(&Key::from_parts(UNPRICED_TOKENS, labels), metadata))
        };
        Counter::from_arc(Arc::new(PricedCounter { counter, side, model, target, config: self.config.clone() }))
    }

    fn register_gauge(&self, key: &Key, metadata: &Metadata<'_>) -> Gauge {
        self.inner.register_gauge(key, metadata)
    }

    fn register_histogram(&self, key: &Key, metadata: &Metadata<'_>) -> Histogram {
        self.inner.register_histogram(key, metadata)
    }
}

enum Target {
    Cost(Gauge),
    Unpriced(Counter),
}

struct PricedCounter {
    counter: Counter,
    side: Side,
    model: String,
    target: Target,
    config: Arc<Config>,
}

impl CounterFn for PricedCounter {
    fn increment(&self, tokens: u64) {
        self.counter.increment(tokens);
        if tokens == 0 {
            return;
        }
        let usage = match self.side {
            Side::Input => Usage::new(tokens, 0),
            Side::Output => Usage::new(0, tokens),
        };
        let cost = match &self.target {
            Target::Cost(cost) => cost,
            Target::Unpriced(unpriced) => {
                unpriced.increment(tokens);
                return;
            }
        };
        let Ok(usd) = self.config.prices().cost_usage(&self.model, &usage) else {
            return;
        };
        cost.increment(usd);
        if let Some(budget) = &self.config.budget {
            let ctx = ChargeContext::new().model(self.model.clone()).usage(&usage);
            let outcome = budget.charge_usd_force(usd, ctx);
            if outcome.over_cap && self.config.kill_on_over_cap && !budget.is_killed() {
                budget.kill("LLM spend counted by the metrics CostLayer is over the budget cap");
            }
        }
    }

    fn absolute(&self, value: u64) {
        self.counter.absolute(value);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use metrics_util::debugging::{DebugValue, DebuggingRecorder, Snapshotter};
    use metrics_util::CompositeKey;

    type Row = (CompositeKey, Option<Unit>, Option<SharedString>, DebugValue);

    fn snapshot(snapshotter: &Snapshotter) -> Vec<Row> {
        snapshotter.snapshot().into_vec()
    }

    fn find<'a>(rows: &'a [Row], name: &str, model: &str) -> Option<&'a DebugValue> {
        rows.iter()
            .find(|(k, _, _, _)| {
                k.key().name() == name && k.key().labels().any(|l| l.key() == "model" && l.value() == model)
            })
            .map(|(_, _, _, v)| v)
    }

    fn gauge(rows: &[Row], name: &str, model: &str) -> f64 {
        match find(rows, name, model) {
            Some(DebugValue::Gauge(g)) => g.into_inner(),
            other => panic!("no gauge {name} for {model}: {other:?}"),
        }
    }

    #[test]
    fn prices_swiftide_counters_and_keeps_the_originals() {
        let inner = DebuggingRecorder::new();
        let snapshotter = inner.snapshotter();
        let recorder = CostLayer::swiftide().layer(inner);
        metrics::with_local_recorder(&recorder, || {
            for _ in 0..2 {
                metrics::counter!("swiftide.usage.prompt_tokens", "model" => "gpt-4o", "pipeline" => "docs")
                    .increment(5_000);
                metrics::counter!("swiftide.usage.completion_tokens", "model" => "gpt-4o", "pipeline" => "docs")
                    .increment(500);
                metrics::counter!("swiftide.usage.total_tokens", "model" => "gpt-4o", "pipeline" => "docs")
                    .increment(5_500);
            }
        });
        let rows = snapshot(&snapshotter);
        assert!(find(&rows, UNPRICED_TOKENS, "gpt-4o").is_none());
        assert_eq!(find(&rows, "swiftide.usage.prompt_tokens", "gpt-4o"), Some(&DebugValue::Counter(10_000)));
        assert_eq!(find(&rows, "swiftide.usage.completion_tokens", "gpt-4o"), Some(&DebugValue::Counter(1_000)));
        assert_eq!(find(&rows, "swiftide.usage.total_tokens", "gpt-4o"), Some(&DebugValue::Counter(11_000)));
        // gpt-4o: $2.50/M in, $10/M out; total_tokens is not priced twice
        assert!((gauge(&rows, "swiftide.usage.cost_usd", "gpt-4o") - 0.035).abs() < 1e-9);
        // the cost series carries the counter's other labels
        let cost = rows.iter().find(|(k, _, _, _)| k.key().name() == "swiftide.usage.cost_usd").map(|(k, ..)| k);
        assert!(cost.is_some_and(|k| k.key().labels().any(|l| l.key() == "pipeline" && l.value() == "docs")));
    }

    #[test]
    fn separate_series_per_model() {
        let inner = DebuggingRecorder::new();
        let snapshotter = inner.snapshotter();
        let recorder = CostLayer::swiftide().layer(inner);
        metrics::with_local_recorder(&recorder, || {
            metrics::counter!("swiftide.usage.prompt_tokens", "model" => "gpt-4o").increment(1_000_000);
            metrics::counter!("swiftide.usage.prompt_tokens", "model" => "gpt-4o-mini").increment(1_000_000);
        });
        let rows = snapshot(&snapshotter);
        assert!((gauge(&rows, "swiftide.usage.cost_usd", "gpt-4o") - 2.50).abs() < 1e-9);
        assert!((gauge(&rows, "swiftide.usage.cost_usd", "gpt-4o-mini") - 0.15).abs() < 1e-9);
    }

    #[test]
    fn unknown_models_and_unlabelled_counters_are_not_priced() {
        let inner = DebuggingRecorder::new();
        let snapshotter = inner.snapshotter();
        let budget = Arc::new(AgentBudget::builder("a", 1.0).build().unwrap());
        let recorder = CostLayer::swiftide().budget(budget.clone()).layer(inner);
        metrics::with_local_recorder(&recorder, || {
            metrics::counter!("swiftide.usage.prompt_tokens", "model" => "my-local-llama").increment(700);
            metrics::counter!("swiftide.usage.prompt_tokens").increment(900);
        });
        let rows = snapshot(&snapshotter);
        assert_eq!(find(&rows, UNPRICED_TOKENS, "my-local-llama"), Some(&DebugValue::Counter(700)));
        assert!(find(&rows, "swiftide.usage.cost_usd", "my-local-llama").is_none());
        assert!(!rows.iter().any(|(k, ..)| k.key().name() == "swiftide.usage.cost_usd"));
        assert!(!rows.iter().any(|(k, ..)| k.key().name() == UNPRICED_TOKENS && k.key().labels().count() == 0));
        assert_eq!(budget.spent_usd(), 0.0);
    }

    #[test]
    fn own_prices_and_counter_names() {
        let inner = DebuggingRecorder::new();
        let snapshotter = inner.snapshotter();
        let table = PriceTable::new().with_price(crate::ModelPrice::new("my-local-llama", "self-hosted", 1.0, 2.0));
        let recorder = CostLayer::new()
            .input_tokens("gen_ai.input_tokens")
            .output_tokens("gen_ai.output_tokens")
            .model_label("gen_ai.request.model")
            .price_table(table)
            .layer(inner);
        metrics::with_local_recorder(&recorder, || {
            metrics::counter!("gen_ai.input_tokens", "gen_ai.request.model" => "my-local-llama").increment(1_000_000);
            metrics::counter!("gen_ai.output_tokens", "gen_ai.request.model" => "my-local-llama").increment(500_000);
        });
        let rows = snapshot(&snapshotter);
        let usd = rows.iter().find(|(k, ..)| k.key().name() == "llm.usage.cost_usd").map(|(.., v)| v);
        match usd {
            Some(DebugValue::Gauge(g)) => assert!((g.into_inner() - 2.0).abs() < 1e-9),
            other => panic!("cost gauge missing: {other:?}"),
        }
    }

    #[test]
    fn books_on_the_budget_and_kills_it_over_the_cap() {
        let budget = Arc::new(AgentBudget::builder("indexer", 0.03).build().unwrap());
        let recorder = CostLayer::swiftide().budget(budget.clone()).layer(DebuggingRecorder::new());
        metrics::with_local_recorder(&recorder, || {
            metrics::counter!("swiftide.usage.prompt_tokens", "model" => "gpt-4o").increment(10_000);
        });
        assert!((budget.spent_usd() - 0.025).abs() < 1e-9);
        assert!(!budget.is_killed());
        metrics::with_local_recorder(&recorder, || {
            metrics::counter!("swiftide.usage.completion_tokens", "model" => "gpt-4o").increment(1_000);
        });
        // already spent, so it is booked past the cap, then the budget is stopped
        assert!((budget.spent_usd() - 0.035).abs() < 1e-9);
        assert!(budget.is_killed());
        assert!(budget.reserve(0.001).is_err());
    }

    #[test]
    fn kill_on_cap_false_leaves_the_kill_switch_alone() {
        let budget = Arc::new(AgentBudget::builder("indexer", 0.01).build().unwrap());
        let recorder = CostLayer::swiftide().budget(budget.clone()).kill_on_cap(false).layer(DebuggingRecorder::new());
        metrics::with_local_recorder(&recorder, || {
            metrics::counter!("swiftide.usage.prompt_tokens", "model" => "gpt-4o").increment(10_000);
        });
        assert!(budget.spent_usd() > budget.limit_usd());
        assert!(!budget.is_killed());
    }

    #[test]
    fn describes_the_cost_gauge() {
        let inner = DebuggingRecorder::new();
        let snapshotter = inner.snapshotter();
        let recorder = CostLayer::swiftide().layer(inner);
        metrics::with_local_recorder(&recorder, || {
            metrics::describe_counter!("swiftide.usage.prompt_tokens", "token usage for the prompt");
            metrics::counter!("swiftide.usage.prompt_tokens", "model" => "gpt-4o").increment(1);
        });
        let rows = snapshot(&snapshotter);
        let described =
            rows.iter().find(|(k, ..)| k.key().name() == "swiftide.usage.cost_usd").and_then(|(_, _, d, _)| d.clone());
        assert!(described.is_some_and(|d| d.contains("USD")));
    }
}
