# llm-budget

Hard spend limits for LLM agents in Rust: every call is charged against a per-agent dollar cap and refused once the cap would be crossed, so a runaway loop stops instead of running up the bill.

[![crates.io](https://img.shields.io/crates/v/llm-budget.svg)](https://crates.io/crates/llm-budget)
[![docs.rs](https://img.shields.io/docsrs/llm-budget)](https://docs.rs/llm-budget)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/Mattbusel/llm-budget/blob/main/LICENSE)
[![CI](https://github.com/Mattbusel/llm-budget/actions/workflows/ci.yml/badge.svg)](https://github.com/Mattbusel/llm-budget/actions/workflows/ci.yml)
![MSRV 1.85](https://img.shields.io/badge/MSRV-1.85-blue.svg)

Give each agent a budget in US dollars, put a fleet-wide total on top, and charge every model call to it. You get a typed error the moment a call would go over, a kill switch, and an automatic step down to a cheaper model tier when an agent is running low. Prices come from a built-in table of 232 real models (OpenAI, Anthropic, Google Gemini, Mistral, DeepSeek, xAI).

## Install

```bash
cargo add llm-budget
cargo add tokio --features macros,rt-multi-thread
```

Optional features: `tokens` (count tokens from text), `ratelimit` (tokens and requests per minute), `tracing`, `metrics`, `schemars`. See the table below.

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

The same program is in `examples/quickstart.rs`: `cargo run --example quickstart`.

## Why this and not something else

- **Provider SDKs and gateways** (async-openai, genai, rig, litellm-rs) report usage after the fact or enforce budgets in a separate server process. `llm-budget` is a small library you call before each request, inside your own process, with no server and no network.
- **Rate limiters** such as [governor](https://crates.io/crates/governor) cap how *fast* you spend, not how *much*. Use both: the `ratelimit` feature wraps governor for tokens-per-minute limits, next to the dollar caps.
- **A `Mutex<f64>` you write yourself** works until two tasks race or a float rounds a tiny charge to zero. Here spend is an atomic nano-dollar counter: the cap cannot be crossed under concurrency, and a 1-token call to a cheap model still counts.

## Already using another crate?

`llm-budget` does not make HTTP calls, so it sits next to whatever client you have:

- **async-openai**: turn on the `async-openai` feature and charge the response's usage directly:
  ```rust,ignore
  let resp = client.chat().create(request).await?;
  if let Some(u) = &resp.usage {
      agent.charge_usage("gpt-4o", &llm_budget::Usage::from(u), None, None).await?;
  }
  ```
- **reqwest, genai, rig, or any other client**: pass the raw response body (or the usage object you have serialised) to `Usage::from_json`; it understands the OpenAI, Anthropic and Gemini formats. See `examples/provider_usage.rs`.
- **governor**: the `ratelimit` feature uses it for tokens-per-minute limits.
- **tracing / metrics**: turn on the features of the same name to get events and Prometheus-style counters for every decision.
- **LiteLLM**: the price table is LiteLLM's; `PriceTable::from_litellm_json` reads the same file a LiteLLM proxy uses.

## Features

| Feature | Default | What it adds | Extra dependencies |
|---|---|---|---|
| (none) | yes | Budgets, fleet caps, kill switch, tier cascade, price table, JSON persistence | tokio (`sync`, `fs`), serde, dashmap, thiserror |
| `tokens` | no | `tokens::count_tokens`, `cost_of_text`, `AgentBudget::charge_model_text`: token counts from text with OpenAI's tokenizers | [tiktoken-rs](https://crates.io/crates/tiktoken-rs) |
| `ratelimit` | no | `ratelimit::TokenRateLimiter`: tokens-per-minute and requests-per-minute limits with a retry-after time | [governor](https://crates.io/crates/governor) |
| `tracing` | no | A `tracing` event for every accepted or refused charge, downgrade and kill | [tracing](https://crates.io/crates/tracing) |
| `metrics` | no | `llm_budget_charges_total`, `llm_budget_spent_usd`, `llm_budget_remaining_usd`, `llm_budget_downgrades_total` through the metrics facade (Prometheus and other exporters) | [metrics](https://crates.io/crates/metrics) |
| `async-openai` | no | `From<&CompletionUsage> for Usage`, so an async-openai response can be charged directly | [async-openai](https://crates.io/crates/async-openai) types only (no HTTP client) |
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

- **`AgentBudget`**: a per-agent cap. Charge from the provider's own usage report (`charge_usage`, the most accurate), by model name and token counts (`charge_model`), by exact dollar amount (`charge_usd`, for when your provider reports the cost), or by tier estimate (`charge`). A charge that would cross the cap is refused with `BudgetError::AgentExhausted` and nothing is recorded.
- **`Usage`**: reads the `usage` block of an OpenAI (Chat Completions or Responses), Anthropic or Gemini response, whole body or just the block, and normalises the different ways they count prompt-cache tokens, so cached tokens are charged at the cache price and never twice.
- **`PriceTable`**: real input, output and prompt-cache prices for 232 chat models, generated from [LiteLLM's maintained price list](https://github.com/BerriAI/litellm) (MIT). `pricing::SNAPSHOT_DATE` says when. Load a newer copy of that JSON file at runtime with `PriceTable::from_litellm_json`, or add your own models with `insert`.
- **`FleetAllocator`**: registers agents against a fleet total, refuses registrations that would oversubscribe it, tracks real fleet spend, moves unspent budget between agents (`reallocate`) and returns it when an agent is removed (`remove_agent`).
- **Kill switch**: `kill(reason)` on one agent or `kill_all` on the fleet. The reason is returned in the error.
- **Downgrade cascade** (`ModelTier`, `ModelCascade`): when the remaining fraction drops to the threshold, the agent steps down one tier and the charge tells you.
- **Lock-free accounting**: spend is an atomic counter of nano-dollars updated with compare-and-swap, so many tasks can charge at once without a mutex and the cap is never exceeded.
- **Audit trail**: every accepted charge is kept as a `SpendRecord` (agent, task, session, model, tokens, cost, time).
- **`BudgetStore`**: saves fleet snapshots as one JSON file per session (written atomically, session ids checked so they cannot point outside the folder).

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

Prices change. Either regenerate the built-in table (`python scripts/update_prices.py`, then rebuild) or load the live file when your program starts:

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

`benches/vs_alternatives.rs` charges 1 micro-dollar repeatedly against a limit that never refuses, so it measures bookkeeping only. Measured on an Intel i7-13700KF (Windows 11, release build, other work running on the machine, so treat the numbers as rough):

| Implementation | 1 thread, per charge | 8 threads x 10,000 charges (includes thread start) |
|---|---|---|
| `llm-budget` 0.2 (`AtomicBudget`, compare-and-swap) | 6.9 ns | 2.3 ms |
| `llm-budget` 0.1.1 (add, then roll back if over) | 4.9 ns | 1.5 ms |
| `std::sync::Mutex<u64>` check-and-add | 9.6 ns | 6.2 ms |
| `governor` 0.10 `check_n` (rate limiter, for scale) | 30.9 ns | 4.1 ms |

0.1.1 is faster because it does not retry, but it can refuse a charge that fits when another thread's larger charge is being rolled back at the same moment (see the CHANGELOG); 0.2 trades about 2 ns per charge for never doing that. Either way the cost is noise next to an HTTP call to a model. Run `cargo bench --bench vs_alternatives` to reproduce.

## Limitations

- The fleet total limits what can be handed out to agents; spending is enforced per agent.
- Charges made directly on an `AgentBudget` you got from `get_agent` are not added to the fleet's spend counter; charge through the fleet for that.
- Gemini models with long-context surcharges are priced at their base rate.

Run the tests with `cargo test --all-features`. Contributions are welcome, see [CONTRIBUTING.md](https://github.com/Mattbusel/llm-budget/blob/main/CONTRIBUTING.md).

## License

MIT, see [LICENSE](https://github.com/Mattbusel/llm-budget/blob/main/LICENSE).

Source, issues and stars: [GitHub](https://github.com/Mattbusel/llm-budget). Mirror and release downloads: [GitLab](https://gitlab.com/mattbusel/llm-budget). More open-source Rust from the same author: [tensorust.com/open-source](https://tensorust.com/open-source/).
