# llm-budget: AI agents have loop limits, not spending limits

[![crates.io](https://img.shields.io/crates/v/llm-budget.svg)](https://crates.io/crates/llm-budget)
[![docs.rs](https://img.shields.io/docsrs/llm-budget)](https://docs.rs/llm-budget)
[![CI](https://github.com/Mattbusel/llm-budget/actions/workflows/ci.yml/badge.svg)](https://github.com/Mattbusel/llm-budget/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/Mattbusel/llm-budget/blob/main/LICENSE)

Almost no agent framework ships a hard dollar cap. They cap loops and iterations, and their guardrails cover prompt injection, PII and toxicity, not money ([AutoAgents #320](https://github.com/liquidos-ai/AutoAgents/pull/320), [langchain-rust #405](https://github.com/Abraxas-365/langchain-rust/pull/405)). Where a cap exists it trusts a price table: in a 9.3K-star framework every newer model fell back to a $1/$3 default, so the budget saw less than a fifth of what a gpt-5.5 call really cost ([PraisonAI #6433](https://github.com/MervinPraison/PraisonAI/pull/6433)). llm-budget prices each call from a dated, source-cited table, reserves the worst case in dollars before the call, and refuses the call that would not fit, or that it cannot price.

- Refuses a call that would go over the cap.
- Caps a whole fleet of agents.
- Stops every agent with one kill switch.
- Steps down to a cheaper model when money runs low.

## Fail closed, on purpose

A soft limit records the overspend after it has happened. Since 0.10 llm-budget refuses every call it cannot price exactly, by default:

| What other trackers do | What llm-budget does | Loud opt-out (warns on every call) |
|---|---|---|
| Unknown model priced at $0 or a $1/$3 default | `UnknownModel`, call refused | `allow_unpriced()` |
| Cached tokens billed as plain input | `CachePriceMissing` unless the cache rate is listed | `allow_unpriced()` |
| Price table pasted into code, never updated | `StalePrice` once a price is 90 days old | `allow_stale_prices()`, `max_price_age_days(n)` |
| Check spend, make the call, book it after | the call is refused before it is sent | `unsafe_soft_limit()` |
| A budget with no cap | does not build | none |

Prices come from [llm-prices](https://crates.io/crates/llm-prices): every row cites the provider's pricing page and the date it was checked.

## How much would a soft limit have let through?

Point the audit at a JSON-lines log of your calls (`model` plus the provider's `usage` block, and optionally `booked_usd`, what your tracker booked):

```bash
cargo install llm-budget
llm-budget audit calls.jsonl --cap 1
```

```text
Soft limit (checks spend before each call, books after, $0 for unpriced models):
  stopped the run at call 10, after letting 9 calls through
  tracker booked $1.0350; those calls really cost $5.8500
  MONEY PAST THE CAP: $4.8500 (5.85x the cap); real spend crossed $1.00 at call 2, then 7 more calls ran

Hard cap, fail closed (llm-budget default):
  admitted 1 calls, spent $0.6500 of $1.00
```

That log is the [PraisonAI](https://github.com/MervinPraison/PraisonAI/pull/6433) pattern: gpt-5.5 calls booked at a $1/$3 default. The exit status is 2 when money got past the cap, so it can gate CI. `--json` for machines.

## Install

```bash
cargo add llm-budget
cargo add tokio --features macros,rt-multi-thread
```

## Use

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

[Docs](https://docs.rs/llm-budget) · [crates.io](https://crates.io/crates/llm-budget)

More: [docs/DETAILS.md](https://github.com/Mattbusel/llm-budget/blob/main/docs/DETAILS.md)

Adding it to another project (features, MSRV, stability, the five-line integration): [docs/INTEGRATING.md](https://github.com/Mattbusel/llm-budget/blob/main/docs/INTEGRATING.md)

MIT license, see [LICENSE](https://github.com/Mattbusel/llm-budget/blob/main/LICENSE).
