# llm-budget: AI agents have loop limits, not spending limits

[![crates.io](https://img.shields.io/crates/v/llm-budget.svg)](https://crates.io/crates/llm-budget)
[![docs.rs](https://img.shields.io/docsrs/llm-budget)](https://docs.rs/llm-budget)
[![CI](https://github.com/Mattbusel/llm-budget/actions/workflows/ci.yml/badge.svg)](https://github.com/Mattbusel/llm-budget/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/Mattbusel/llm-budget/blob/main/LICENSE)

Almost no agent framework ships a hard dollar cap. They cap loops and iterations, and their guardrails cover prompt injection, PII and toxicity, not money ([AutoAgents #320](https://github.com/liquidos-ai/AutoAgents/pull/320), [langchain-rust #405](https://github.com/Abraxas-365/langchain-rust/pull/405)). Where a cap exists it trusts a price table: in a 9.3K-star framework every newer model fell back to a $1/$3 default, so the budget saw less than a fifth of what a gpt-5.5 call really cost ([PraisonAI #6433](https://github.com/MervinPraison/PraisonAI/pull/6433)). llm-budget prices 232 models, reserves the worst case in dollars before each call, and refuses the call that would not fit.

- Refuses a call that would go over the cap.
- Caps a whole fleet of agents.
- Stops every agent with one kill switch.
- Steps down to a cheaper model when money runs low.

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
