# llm-budget

[![CI](https://github.com/Mattbusel/llm-budget/actions/workflows/ci.yml/badge.svg)](https://github.com/Mattbusel/llm-budget/actions/workflows/ci.yml)
[![crates.io](https://img.shields.io/crates/v/llm-budget.svg)](https://crates.io/crates/llm-budget)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Hard spend limits for LLM agents in Rust: every call is charged against a per-agent budget and refused once the cap is hit, instead of alerting after the bill arrives.

When you run many agents at once, one runaway loop can burn a day's budget in minutes. `llm-budget` gives each agent a hard USD cap, a fleet-wide cap on top, a kill switch, and an automatic model downgrade (Premium to Standard to Economy to Local) as an agent's remaining budget gets low.

## Features

- **`AgentBudget`**: per-agent USD limit. `charge(input_tokens, output_tokens, ..)` estimates the cost from the agent's current model tier and rejects the charge with `BudgetError::AgentExhausted` if it would cross the limit.
- **Automatic downgrade cascade** (`ModelCascade`, `ModelTier`): when the remaining fraction of the budget drops below a threshold, the agent steps down one tier and `charge` tells you which tier to use next.
- **Kill switch**: `kill(reason)` on one agent, or `FleetAllocator::kill_all` for the whole fleet; later charges fail with `KillSwitchActivated`.
- **`FleetAllocator`**: registers agents against a fleet-wide cap and refuses allocations that would oversubscribe it (`AllocationFailed`).
- **Lock-free accounting**: spend is stored as atomic micro-dollars (`AtomicBudget`), so concurrent charges from many tasks are safe without a mutex on the hot path.
- **Audit trail**: every accepted charge is kept as a `SpendRecord` (agent, task, session, tier, tokens, cost, timestamp).
- **Persistence**: `BudgetStore` saves and loads fleet snapshots as JSON files per session.
- **Typed errors** only: the crate denies `unwrap`, `expect` and `panic` via Clippy lints.

## Quick start

```bash
cargo add llm-budget
cargo add tokio --features full
```

```rust
use llm_budget::{AgentBudget, BudgetError, FleetAllocator, ModelTier};

#[tokio::main]
async fn main() -> Result<(), BudgetError> {
    // One agent: $5 cap, starts on the Premium tier,
    // downgrades when less than 20% of the budget is left.
    let agent = AgentBudget::new("researcher", 5.0, ModelTier::Premium, 0.2)?;

    let (total_spent, downgraded_to) = agent.charge(1_200, 400, None, None).await?;
    println!("spent ${total_spent:.4}, now on {:?}", agent.current_tier().await);
    if let Some(tier) = downgraded_to {
        println!("switch to a {tier:?} model");
    }

    // A fleet: $100 shared across agents, each with its own cap.
    let fleet = FleetAllocator::new(100.0, ModelTier::Standard, 0.2)?;
    fleet.register_agent("planner", 40.0)?;
    fleet.register_agent("coder", 40.0)?;
    assert!(fleet.register_agent("extra", 40.0).is_err()); // would exceed $100

    fleet.charge_agent("coder", 2_000, 800, Some("task-42".into()), None).await?;

    // Emergency stop.
    fleet.kill_all("runaway loop detected");
    assert!(fleet.charge_agent("coder", 10, 10, None, None).await.is_err());

    for snap in fleet.fleet_snapshot().await {
        println!("{} spent ${:.4} of ${:.2}", snap.agent_id, snap.spent_usd, snap.limit_usd);
    }
    Ok(())
}
```

Saving state between runs:

```rust
use llm_budget::{BudgetError, BudgetStore, FleetAllocator};

async fn save(fleet: &FleetAllocator) -> Result<(), BudgetError> {
    let store = BudgetStore::new("./budget-state");
    store.save("session-1", fleet.fleet_snapshot().await).await?;
    let restored = store.load("session-1").await?;
    println!("{} agents restored", restored.snapshots.len());
    Ok(())
}
```

## How it works

| File | What it holds |
|---|---|
| `src/budget/mod.rs` | `AtomicBudget` (micro-dollar atomics), `BudgetEnvelope` (agent/task/session scope, kill switch, audit log), `AgentBudget`, `BudgetSnapshot` |
| `src/model.rs` | `ModelTier` with built-in per-1K-token prices, `ModelCascade` downgrade logic |
| `src/fleet/mod.rs` | `FleetBudget` (fleet-wide cap), `FleetAllocator` (agent registry on a `DashMap`) |
| `src/persistence/mod.rs` | `BudgetStore`, JSON snapshots on disk with an in-memory cache |
| `src/error.rs` | `BudgetError` |

A charge is a single `fetch_add` on an `AtomicU64`; if the new total crosses the limit it is rolled back with `fetch_sub` and the call returns an error, so the limit is never exceeded.

## Status and limitations

Version 0.1, early. Things to know before relying on it:

- Costs come from four fixed tiers with approximate prices (`ModelTier::cost_per_1k_input/output`), not from per-model price tables.
- `FleetAllocator::reallocate` validates the request but does not move budget between agents yet.
- Fleet-level spend recorded by `charge_agent` is approximate; per-agent limits are the exact ones.

Run the tests and benchmarks with:

```bash
cargo test
cargo bench
```

## License

MIT, see [LICENSE](LICENSE).

---

Part of a set of Rust crates for LLM agents, see [rust-crates](https://github.com/Mattbusel/rust-crates).
