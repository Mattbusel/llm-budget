# llm-budget: details

Everything cut from the main [README](../README.md).

<!-- outcome-block -->
**Stops a runaway agent before it runs up your bill. Each call is charged to a dollar cap and refused if it would go over.**

Provider alerts come after the money is gone. This refuses the call first. Prices for 232 real models built in.

If it doesn't do what this README says, [open an issue](https://github.com/Mattbusel/llm-budget/issues) and I'll fix it.
<!-- outcome-block -->

Give each agent a dollar budget. Put a fleet total on top. Charge every model call to it. You get a typed error when a call would go over, a kill switch, and a step down to a cheaper model when an agent runs low. Prices for 232 models (OpenAI, Anthropic, Google Gemini, Mistral, DeepSeek, xAI).

## Install

```bash
cargo add llm-budget
cargo add tokio --features macros,rt-multi-thread
```

Optional features: `tokens`, `ratelimit`, `tracing`, `metrics`, `schemars`, `async-openai`, `rig`. Table below.

## In ten lines

```rust
use llm_budget::{AgentBudget, BudgetError, ModelTier};

#[tokio::main]
async fn main() -> Result<(), BudgetError> {
    let agent = AgentBudget::new("researcher", 0.05, ModelTier::Standard, 0.2)?; // $0.05 cap
    agent.charge_model("gpt-4o", 10_000, 500, None, None).await?;               // $0.03, fine
    match agent.charge_model("gpt-4o", 10_000, 500, None, None).await {        // would reach $0.06
        Err(BudgetError::AgentExhausted { spent, limit, .. }) => println!("stopped at ${spent} of ${limit}"),
        other => panic!("the cap should have stopped this call: {other:?}"),
    }
    Ok(())
}
```

## Quickstart

```rust
use llm_budget::{AgentBudget, BudgetError, FleetAllocator, ModelTier};

#[tokio::main]
async fn main() -> Result<(), BudgetError> {
    // One agent with a $5 cap. When 20% or less is left it is told to
    // move to a cheaper tier (Premium, Standard, Economy, Local).
    let agent = AgentBudget::new("researcher", 5.0, ModelTier::Premium, 0.2)?;

    // Charge a real call at the model's real price:
    // gpt-4o, 1,200 prompt tokens and 400 completion tokens.
    let receipt = agent.charge_model("gpt-4o", 1_200, 400, None, None).await?;
    println!("that call cost ${:.6}, ${:.4} left", receipt.cost_usd, agent.remaining_usd());

    // A fleet: $100 shared by several agents, each with its own cap.
    let fleet = FleetAllocator::new(100.0, ModelTier::Standard, 0.2)?;
    fleet.register_agent("planner", 40.0)?;
    fleet.register_agent("coder", 40.0)?;
    assert!(fleet.register_agent("extra", 40.0).is_err()); // would exceed $100

    fleet.charge_agent_model("coder", "claude-sonnet-4-5", 2_000, 800, None, None).await?;
    fleet.reallocate("planner", "coder", 10.0).await?; // move unspent budget

    // Emergency stop: every later charge fails with KillSwitchActivated.
    fleet.kill_all("runaway loop detected");
    assert!(fleet.charge_agent_model("coder", "gpt-4o", 10, 10, None, None).await.is_err());

    for snap in fleet.fleet_snapshot().await {
        println!("{} spent ${:.4} of ${:.2}", snap.agent_id, snap.spent_usd, snap.limit_usd);
    }
    Ok(())
}
```

Also in `examples/quickstart.rs`: `cargo run --example quickstart`.

## Why this

- **SDKs and gateways** (async-openai, genai, rig, litellm-rs) report usage after the fact, or need a separate server. This is a library you call before each request. No server, no network.
- **Rate limiters** like [governor](https://crates.io/crates/governor) cap how *fast* you spend, not how *much*. The `ratelimit` feature wraps governor, so you can use both.
- **Your own `Mutex<f64>`** works until two tasks race or a float rounds a tiny charge to zero. Here spend is an atomic nano-dollar counter. The cap holds under concurrency, and a 1-token call still counts.

## Already using another crate?

It makes no HTTP calls, so it works next to any client:

- **async-openai**: turn on `async-openai` and charge the response's usage:
  ```rust,ignore
  let resp = client.chat().create(request).await?;
  if let Some(u) = &resp.usage {
      agent.charge_usage("gpt-4o", &llm_budget::Usage::from(u), None, None).await?;
  }
  ```
- **rig** (rig-agent 0.44): turn on `rig` and attach `BudgetHook`. Each model turn is charged at the real price (cache reads included). The run stops with `Agent 'researcher' budget exhausted: ...` at the turn that would cross the cap. A spent budget stops before calling the provider:
  ```rust,ignore
  use std::sync::Arc;
  use llm_budget::{rig::BudgetHook, AgentBudget, ModelTier};
  use rig_agent::AgentBuilder;

  let budget = Arc::new(AgentBudget::new("researcher", 0.50, ModelTier::Standard, 0.2)?); // $0.50 per agent
  let agent = AgentBuilder::new(model) // any rig completion model, here OpenAI gpt-4o
      .preamble("You are a research assistant.")
      .add_hook(BudgetHook::new(budget.clone(), "gpt-4o"))
      .build();
  let answer = agent.prompt("Summarise these 40 papers").max_turns(30).await; // Err once the cap is hit
  println!("spent ${:.4}", budget.spent_usd());
  ```
  Share one `Arc<AgentBudget>` for a common cap. `rig` needs Rust 1.95. The rest keeps MSRV 1.85.
- **reqwest, genai, or any client**: pass the raw response body (or usage object) to `Usage::from_json`. It reads OpenAI, Anthropic and Gemini formats. See `examples/provider_usage.rs`.
- **governor**: `ratelimit` uses it for tokens-per-minute limits.
- **tracing / metrics**: same-named features give events and Prometheus-style counters for every decision.
- **LiteLLM**: the price table is LiteLLM's. `PriceTable::from_litellm_json` reads the same file.

## Features

| Feature | Default | What it adds | Extra dependencies |
|---|---|---|---|
| (none) | yes | Budgets, fleet caps, kill switch, tier cascade, price table, JSON persistence | tokio (`sync`, `fs`), serde, dashmap, thiserror |
| `tokens` | no | `tokens::count_tokens`, `cost_of_text`, `AgentBudget::charge_model_text`: token counts from text with OpenAI's tokenizers | [tiktoken-rs](https://crates.io/crates/tiktoken-rs) |
| `ratelimit` | no | `ratelimit::TokenRateLimiter`: tokens-per-minute and requests-per-minute limits with a retry-after time | [governor](https://crates.io/crates/governor) |
| `tracing` | no | A `tracing` event for every accepted or refused charge, downgrade and kill | [tracing](https://crates.io/crates/tracing) |
| `metrics` | no | `llm_budget_charges_total`, `llm_budget_spent_usd`, `llm_budget_remaining_usd`, `llm_budget_downgrades_total` through the metrics facade (Prometheus and other exporters) | [metrics](https://crates.io/crates/metrics) |
| `async-openai` | no | `From<&CompletionUsage> for Usage`, so an async-openai response can be charged directly | [async-openai](https://crates.io/crates/async-openai) types only (no HTTP client) |
| `rig` | no | `rig::BudgetHook`, an `AgentHook` for rig-agent 0.44 that charges every model turn and stops the run at the cap, and `From<&rig Usage> for Usage`. Needs Rust 1.95 | [rig-agent](https://crates.io/crates/rig-agent) (no default features) |
| `schemars` | no | JSON Schema for `SpendRecord`, `BudgetSnapshot`, `ChargeReceipt` and the saved state, for non-Rust readers of those files | [schemars](https://crates.io/crates/schemars) |

## Examples

| Example | Shows | Run |
|---|---|---|
| `quickstart` | agents, fleet, real prices, reallocation, kill switch | `cargo run --example quickstart` |
| `provider_usage` | charging from OpenAI, Anthropic and Gemini `usage` blocks with cache discounts | `cargo run --example provider_usage` |
| `persist_restore` | saving spend at the end of a run and resuming | `cargo run --example persist_restore` |
| `tokens` | counting tokens from text, cost of a prompt | `cargo run --example tokens --features tokens` |
| `rate_limit` | a TPM limit and a dollar cap together, with tracing output | `cargo run --example rate_limit --features ratelimit,tracing` |

## What you get

- **`AgentBudget`**: a per-agent cap. Charge from the provider's usage report (`charge_usage`, most accurate), model and token counts (`charge_model`), exact dollars (`charge_usd`), or tier estimate (`charge`). Over the cap: `BudgetError::AgentExhausted`, nothing recorded.
- **`Usage`**: reads the `usage` block from OpenAI (Chat Completions or Responses), Anthropic or Gemini. Cached tokens are charged at the cache price, never twice.
- **`PriceTable`**: input, output and cache prices for 232 chat models, from [LiteLLM's price list](https://github.com/BerriAI/litellm) (MIT). `pricing::SNAPSHOT_DATE` says when. Load a newer file with `PriceTable::from_litellm_json`, or add models with `insert`.
- **`FleetAllocator`**: registers agents under a fleet total and refuses oversubscription. Tracks fleet spend. Moves unspent budget (`reallocate`) and returns it on `remove_agent`.
- **Kill switch**: `kill(reason)` on one agent, `kill_all` on the fleet. The reason comes back in the error.
- **Downgrade cascade** (`ModelTier`, `ModelCascade`): at the threshold, the agent steps down one tier and the charge tells you.
- **Lock-free**: spend is an atomic nano-dollar counter with compare-and-swap. Many tasks charge at once, no mutex, cap never exceeded.
- **Audit trail**: each charge is a `SpendRecord` (agent, task, session, model, tokens, cost, time).
- **`BudgetStore`**: saves one JSON file per session (atomic writes, session ids checked so they stay in the folder).

### Counting tokens from text (feature `tokens`)

```rust,ignore
use llm_budget::tokens::count_tokens;

let n = count_tokens("gpt-4o", "Hello, world!");
assert_eq!(n.tokens, 4);   // exact: gpt-4o's own tokenizer
let c = count_tokens("claude-sonnet-4-5", "Hello, world!");
assert!(!c.exact);         // Anthropic's tokenizer is not public, so this is an estimate
```

`cargo run --example tokens --features tokens` runs this.

### Keeping prices current

Prices change. Regenerate the table (`python scripts/update_prices.py`, then rebuild) or load the live file at startup:

```rust,no_run
let json = std::fs::read_to_string("model_prices_and_context_window.json").expect("download it first");
let table = llm_budget::PriceTable::from_litellm_json(&json).expect("valid price JSON");
println!("{} models", table.len());
```

Then charge with `AgentBudget::charge_model_with(&table, ...)`.

## How it works

| File | What it holds |
|---|---|
| `src/budget/mod.rs` | `AtomicBudget` (nano-dollar atomics), `BudgetEnvelope` (scope, kill switch, audit log), `AgentBudget`, `ChargeReceipt` |
| `src/pricing.rs`, `src/prices_data.rs` | `PriceTable`, `ModelPrice`, the generated price table |
| `src/usage.rs` | `Usage`, parsing provider usage blocks |
| `src/tokens.rs` | token counting (feature `tokens`) |
| `src/ratelimit.rs` | `TokenRateLimiter` (feature `ratelimit`) |
| `src/observe.rs` | tracing events and metrics (features `tracing`, `metrics`) |
| `src/model.rs` | `ModelTier` with rough per-tier prices, `ModelCascade` |
| `src/fleet/mod.rs` | `FleetBudget`, `FleetAllocator` |
| `src/persistence/mod.rs` | `BudgetStore` |
| `src/error.rs` | `BudgetError` |

## Performance

`benches/vs_alternatives.rs` charges 1 micro-dollar over and over against a limit that never refuses. Bookkeeping only. Intel i7-13700KF, Windows 11, release build, other work running, so numbers are rough:

| Implementation | 1 thread, per charge | 8 threads x 10,000 charges (includes thread start) |
|---|---|---|
| `llm-budget` 0.2 (`AtomicBudget`, compare-and-swap) | 6.9 ns | 2.3 ms |
| `llm-budget` 0.1.1 (add, then roll back if over) | 4.9 ns | 1.5 ms |
| `std::sync::Mutex<u64>` check-and-add | 9.6 ns | 6.2 ms |
| `governor` 0.10 `check_n` (rate limiter, for scale) | 30.9 ns | 4.1 ms |

0.1.1 is faster because it doesn't retry, but it can wrongly refuse a charge during another thread's rollback (see the CHANGELOG). 0.2 costs about 2 ns more and never does. Both are noise next to an HTTP call. Reproduce: `cargo bench --bench vs_alternatives`.

## Limitations

- The fleet total limits what agents get. Spending is enforced per agent.
- Charges on an `AgentBudget` from `get_agent` skip the fleet's spend counter. Charge through the fleet.
- Gemini long-context surcharges are priced at the base rate.

Tests: `cargo test --all-features`. Contributions welcome, see [CONTRIBUTING.md](https://github.com/Mattbusel/llm-budget/blob/main/CONTRIBUTING.md).

## License

MIT, see [LICENSE](https://github.com/Mattbusel/llm-budget/blob/main/LICENSE).

Source, issues and stars: [GitHub](https://github.com/Mattbusel/llm-budget). Mirror and release downloads: [GitLab](https://gitlab.com/mattbusel/llm-budget). More of my Rust: [tensorust.com/open-source](https://tensorust.com/open-source/).
