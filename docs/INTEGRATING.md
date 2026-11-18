# Integrating llm-budget into another project

For maintainers deciding whether to depend on this crate, and for anyone
wiring a hard dollar cap into an existing agent, client, gateway or
framework. Everything here is checked: the snippets below are compiled and
run by `tests/api_surface.rs` and the examples in CI.

## At a glance

| | |
|---|---|
| License | MIT ([LICENSE](../LICENSE)) |
| MSRV | Rust 1.85 (`rust-version` in Cargo.toml), tested in CI. The `rig` feature follows rig-agent's own minimum (1.95+) and `genai` follows genai's. |
| Unsafe code | None: `#![forbid(unsafe_code)]` |
| Panics | Library code does not `unwrap`, `expect` or `panic!` (denied by clippy lints in Cargo.toml); bad input is a `BudgetError`. |
| Network, files, globals | None in the core. No HTTP client, no background threads, no global state. Files only through `BudgetStore` (feature `store`). |
| Async runtime | Not needed. `default-features = false` drops tokio entirely; the `async` charge methods never await anything runtime-specific. |
| Thread safety | Every shared type is `Send + Sync + 'static` (asserted in `tests/api_surface.rs`). Spend is an atomic nano-dollar counter; the cap holds under any number of concurrent callers. |
| Maintainer | Matthew Busel ([Tensorust LLC](https://tensorust.com/open-source/)). Prices are updated as providers change them; issues and PRs on [GitHub](https://github.com/Mattbusel/llm-budget). |

## Add it

```toml
[dependencies]
# Smallest build: budgets, prices, reserve/settle, scopes, context carriers. No tokio.
llm-budget = { version = "0.8", default-features = false }
```

Without default features the crate pulls in serde, serde_json, thiserror,
dashmap and [llm-prices](https://crates.io/crates/llm-prices) (dated,
source-cited prices with no dependencies of its own). Everything heavier is
behind a feature:

| Feature | Default | Adds | Pulls in |
|---|---|---|---|
| `store` | yes | `BudgetStore`: save and restore spend as JSON | tokio (`sync`, `fs`) |
| `tower` | no | `tower_layer::BudgetLayer`: refuse a call over the cap before it is sent, settle reported usage after | tower, http, bytes |
| `tracing-layer` | no | `tracing_layer::CostLayer`: price spans carrying OpenTelemetry GenAI usage fields, optional cap | tracing, tracing-subscriber (`registry`) |
| `metrics-layer` | no | `metrics_layer::CostLayer`: price token counters already emitted through `metrics` | metrics, metrics-util |
| `async-openai` | no | `openai`: reserve from a request, settle from a response or stream's last chunk | async-openai (types only) |
| `genai` | no | `genai::exec_chat` / `exec_chat_stream` under a hold | genai |
| `rig` | no | `rig::BudgetHook` for rig-agent 0.44 to 0.46 | rig-agent (no default features) |
| `tokens` | no | token counts from text (OpenAI tokenizers) | tiktoken-rs |
| `ratelimit` | no | tokens- and requests-per-minute limits | governor |
| `tracing`, `metrics` | no | events and counters for every charge, refusal, downgrade and kill | tracing / metrics |
| `schemars` | no | JSON Schema for the serialized types | schemars |

### As an optional dependency of your crate

If your users should opt in, put it behind a feature of your own:

```toml
[dependencies]
llm-budget = { version = "0.8", default-features = false, optional = true }

[features]
budget = ["dep:llm-budget"]
```

## The five-line integration

Works with any client (reqwest, an SDK, a subprocess). Reserve the worst
case before the call, settle what the provider reported after it:

```rust
use llm_budget::{AgentBudget, Usage};

let budget = AgentBudget::builder("my-agent", 5.00).build()?;              // $5 hard cap
let hold = budget.reserve_for("gpt-4o", &Usage::new(2_000, 500))?;         // refused if it would not fit
let usage = call_model("...");                                              // the real call, with any client
hold.settle_usage("gpt-4o", &usage);                                        // books exactly what was reported
```

- A refused reservation is `BudgetError::AgentExhausted` (or `KillSwitchActivated`,
  `ScopeClosed`, `UnknownModel` / `Unpriced`). The call is never made.
- Dropping a `Reservation` without settling releases the hold, so an error or
  a cancelled future books nothing.
- `Usage::from_json` reads OpenAI (Chat Completions and Responses), Anthropic
  and Gemini usage blocks, cache reads and writes included.
- `budget.record(model, &usage)` books a call you only hear about afterwards.

## Pick the integration point

| Your code has | Use | Example |
|---|---|---|
| Any client, any framework | `AgentBudget::reserve_for` + `Reservation::settle_usage` | `examples/record_usage.rs` |
| A tower stack (hyper, reqwest-middleware, a proxy) | `BudgetLayer::openai_http(budget)` or `BudgetLayer::new(budget, estimate, read_usage)` | `examples/tower_budget.rs` |
| `tracing` spans with `gen_ai.usage.*` fields | `tracing_layer::CostLayer::new().budget(budget)` in the subscriber; the traced code does not change | `examples/rig_tracing_cost.rs` |
| `metrics` token counters | `metrics_layer::CostLayer` around your recorder | `examples/metrics_cost.rs` |
| rig agents | `rig::BudgetHook::new(budget, "gpt-4o")` via `add_hook` | `examples/rig_budget_hook.rs` |
| genai | `genai::exec_chat(&client, &budget, model, req, opts)` | `examples/genai_budget.rs` |
| async-openai | `openai::reserve_request` / `openai::settle_response` | `examples/async_openai_budget.rs` |
| Sub-agents in parallel | `parent.child(id, Limit::Usd(x) or Limit::Fraction(f))` | `examples/fanout.rs`, [budget-scopes.md](budget-scopes.md) |
| Remote agents or MCP tools | `context::BudgetContext::outbound(&scope)` over `a2a`, `baggage` or `mcp` carriers; callee: `AgentBudget::from_context` | `examples/budget_context.rs` |
| Many agents under one total | `FleetAllocator` | `examples/quickstart.rs` |

### Sub-agents

```rust
use llm_budget::{AgentBudget, Limit};

let parent = AgentBudget::builder("orchestrator", 1.00).build()?;
let child = parent.child("researcher", Limit::Usd(0.25))?; // reserved from the parent up front
child.reserve(0.10)?.settle(0.08);
let receipt = child.settle();                               // the unused $0.17 goes back
```

Parallel children can never be admitted for more than the parent has left,
which "pass the remainder down" cannot guarantee (measured in
[budget-scopes.md](budget-scopes.md)). A Python service in the same call
graph can use [agent-budget](https://github.com/Mattbusel/agent-budget),
which reads and writes the same wire format (shared golden fixtures).

## Prices

The built-in table comes from llm-prices (dated, source-cited, long-context
tiers) over a LiteLLM snapshot for other providers. To stay independent of
release cadence:

- `PriceTable::builtin().clone().with_price(ModelPrice::new(..))` adds or
  overrides a model (self-hosted, negotiated rates).
- `PriceTable::from_litellm_json(&json)` loads a newer LiteLLM file at startup.
- `PriceTable::with_hook` plugs in your own lookup.
- `AgentBudgetBuilder::unpriced_policy` decides what happens for a model
  with no price: allow, reserve a fixed amount, or deny.

## Stability

- The public API is everything re-exported at the crate root plus the
  feature modules. Breaking changes go in a minor release while the crate is
  0.x (0.8 to 0.9), are listed in [CHANGELOG.md](../CHANGELOG.md), and come
  with a migration note. Patch releases (0.8.x) do not break code that
  compiles.
- Price data is data: patch releases update prices as providers change them.
  Pin a `PriceTable` of your own if you need fixed numbers in tests.
- The MSRV is raised only in a minor release.
- `__rig-test-utils` is internal and not covered by semver.
