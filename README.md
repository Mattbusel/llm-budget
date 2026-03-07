# llm-budget

Autonomous cost governance primitives — hard budget enforcement across agent fleets.

Track token spend per model, per agent, and per fleet. Block requests before they exceed budget limits. Emit structured cost events for audit trails.

## What's inside

- **BudgetLedger** — per-model token accounting with configurable hard and soft limits
- **FleetGovernor** — aggregate budget across many agents; enforce global spend caps
- **CostEvent** — structured cost records for logging, alerting, and analytics
- **Budget policies** — daily, per-request, per-agent, and rolling-window limits

## Features

- Hard limits that **refuse requests** before they exceed budget (not just alert after)
- Atomic ledger updates — safe under concurrent agent load
- Pluggable cost tables — bring your own per-token pricing for any model

## Quick start

```rust
use llm_budget::{BudgetLedger, BudgetPolicy};

let ledger = BudgetLedger::new(BudgetPolicy {
    daily_usd_limit: 10.0,
    per_request_token_limit: 4096,
});

// Before sending a request
ledger.check_and_reserve("gpt-4o", 1200)?;

// After receiving response
ledger.commit("gpt-4o", 1200, 0.036)?;
```

## Add to your project

```toml
[dependencies]
llm-budget = { git = "https://github.com/Mattbusel/llm-budget" }
```

## Test coverage

```bash
cargo test
```

---

> Used inside [tokio-prompt-orchestrator](https://github.com/Mattbusel/tokio-prompt-orchestrator) -- a production Rust orchestration layer for LLM pipelines. See the full [primitive library collection](https://github.com/Mattbusel/rust-crates).