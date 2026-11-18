# borg budgets priced per model

Feature `borg`. For [borg](https://github.com/joydle/borg) compositions that
mix models: every call is charged by what it cost on the model it actually
used, while borg's `Budget` keeps counting tokens.

API docs: [`llm_budget::borg`](https://docs.rs/llm-budget/latest/llm_budget/borg/index.html)
and [`borg_adapter!`](https://docs.rs/llm-budget/latest/llm_budget/macro.borg_adapter.html).

## Why tokens alone misprice a tree

borg caps a whole composition tree with one token budget. A tree with a
planner on a small model and a reviewer on a frontier model spends tokens
that differ in price by 15x to 100x, so a token cap is either far too loose
for the frontier calls or far too tight for the small ones.

Switching the budget to dollars does not work either: borg's executor admits
parallel branches at `admit_parallel(n, 1_000)` tokens, `BudgetExhausted`
reports `needed` in tokens, and `SubBudget` takes the minimum of its own and
its parent's `remaining()`. All of that assumes token units.

## Reference tokens

Pick a reference model, normally the main model of the tree. Budgets stay in
tokens of that model. A call on any model is charged the number of
reference-model tokens that cost the same, at that call's input/output mix:

```text
charged = tokens_in_call * cost(call's model, usage) / cost(reference, same tokens, no cache)
```

| Call | Charged |
|---|---|
| 1,000 in + 1,000 out on the reference model | 2,000 (exactly its tokens) |
| the same on a model 15x the price | 30,000 |
| the same on a model 20x cheaper | 100 |
| 10,000 in (9,000 cache reads at 10%) on the reference | 1,900 |

So `admit_parallel(n, 1_000)` still means a thousand tokens of the main
model, `BudgetExhausted { needed, remaining }` is tokens on both sides, and
`SubBudget` compares like with like.

## Setup

borg is distributed from GitHub, not crates.io, so llm-budget cannot depend
on it. The `borg_adapter!` macro writes two small types into your crate
instead, compiled against the borg your crate already uses:

```toml
[dependencies]
borg = { git = "https://github.com/joydle/borg", tag = "v1.0.0" }
llm-budget = { version = "0.9", default-features = false, features = ["borg"] }
```

```rust
llm_budget::borg_adapter!(); // PricedBudget and PricedLlm, defined here

use std::sync::Arc;
use llm_budget::borg::BorgPricing;

let pricing = Arc::new(BorgPricing::new("claude-sonnet-4-6")?); // budgets in sonnet tokens
let llm = PricedLlm::shared(&pricing, my_client);              // notes each call's model
let session = PricedBudget::in_memory(&pricing, 200_000);      // 200k sonnet tokens' worth
let ctx = borg::ExecCtx::new(llm, session.clone(), emitter, cancel, "session-1");

// A pool for one composition, also capped by the session:
let round = PricedBudget::sub(&pricing, session.clone(), 50_000);

borg::execute(tree, ctx).await?;
println!("${:.4}", pricing.spent_usd());
```

- `PricedLlm` wraps your `AgentLlmClient`. When a response returns it records
  `(request.model, usage)` for the current thread on the shared `BorgPricing`.
- `PricedBudget` wraps any `Arc<dyn borg::Budget>`. Its `record_usage` takes
  that note, prices the call on the noted model and charges the inner budget
  with `record(reference_tokens)`. `remaining`, `limit`, `can_invoke` and
  `admit_parallel` pass straight through.
- borg's default `BorgAgent::invoke` calls the model and then
  `ctx.budget.record_usage` in the same poll, with no await between them, so
  the note and the charge always meet on the same thread. The note is also
  matched on the exact usage.

## Rules

- Keep a `PricedBudget` outermost on any budget you put in an `ExecCtx`.
  `PricedBudget::sub(&pricing, parent, n)` builds a priced `SubBudget`. A
  bare `borg::sub_budget(priced, n)` would count its own pool in raw tokens.
  Nesting priced pools is fine: an inner one charges its parents through
  `record`, which passes reference tokens through unchanged.
- A charge with no matching note (an agent that overrides `invoke` and
  records on its own, or a client not wrapped in `PricedLlm`) is priced as
  the reference model and counted in `pricing.unattributed_calls()`. Check
  it in tests; it should stay at 0.
- A model the table cannot price is charged its tokens times
  `BorgPricing::unpriced_ratio` (default 1.0) and counted in
  `pricing.unpriced_calls()`.
- Use your own prices with `BorgPricing::with_table(Arc<PriceTable>, reference)`.
  The table is shared, not copied. `BorgPricing::new` uses the built-in table.
- `pricing.units(model, &usage)` returns the charge for a call without
  booking it.

## Tests

`integrations/borg` is a small crate that runs the adapter inside real borg
compositions (borg v1.0.0): a mixed-model sequential and parallel tree, 64
parallel branches on two models across 8 threads, parallel admission
degrading in tokens, and sub-budgets exhausted by price.

```bash
cargo test --manifest-path integrations/borg/Cargo.toml
```
