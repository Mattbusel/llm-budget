# Budget scopes for parallel agents

An orchestrator that hands work to several sub-agents at once needs one
rule to hold: **the children of a delegation together cannot spend more than
the parent has left.** Per-call cost metering does not give you that, and
neither does passing "you have $X remaining" down to each child. This page
shows why, and how child scopes in llm-budget 0.7 enforce it.

## The overshoot

The usual pattern is to tell each delegate how much is left and let it
enforce that figure on its own. When delegates start in parallel, they all
read the same remainder before any of them has spent anything:

```text
parent cap $1.00
  delegate A reads "remaining = $1.00"   delegate B reads "remaining = $1.00"
  A spends $0.80 (under its $1.00)        B spends $0.80 (under its $1.00)
parent spent $1.60
```

Each check passed. The sum did not. Nothing in the per-child view can catch
it, because each child's number was valid when it read it and stale the
moment a sibling spent.

Measured with this crate (both are in the repository and run in CI):

| Setup | What ran | Spent against a $1.00 cap |
|---|---|---|
| `tests/scopes.rs`, `propagated_remainder_overshoots_with_two_parallel_delegates` | 2 delegates in parallel, each told the remainder, four $0.20 calls each | $1.60 (60% over) |
| `examples/fanout.rs`, part 1 | 4 sub-agents in parallel, each told the remainder, eight gpt-4o calls each (reserved at $0.06, settled at $0.052) | $1.664 (66% over), 32 calls |
| `examples/fanout.rs`, part 2 | the same 4 sub-agents, each in `parent.child(.., Limit::Usd(0.25))` | $0.832, 16 calls, $0.000 left held |
| `tests/scopes.rs`, `child_scopes_hold_the_cap_with_two_parallel_delegates` | the 2 delegates with child scopes asking for $0.80 each | $0.80: one got its slice, the other was refused before spending |

Run them yourself:

```bash
cargo run --example fanout
cargo test --test scopes
```

## Reserve, then settle, at the delegation boundary

```rust
use llm_budget::{AgentBudget, Limit, Usage};

let parent = AgentBudget::builder("orchestrator", 1.00).build()?;

// Reserve each slice from the parent before the sub-agent starts.
let research = parent.child("researcher", Limit::Usd(0.60))?;
let writing = parent.child("writer", Limit::Fraction(0.5))?; // half of what is left

// A child is a full AgentBudget: reserve_for / settle_usage, record,
// charge_*, kill, and child() for a grandchild all work on it.
let hold = research.reserve_for("gpt-4o", &Usage::new(20_000, 1_000))?;
// ... make the call ...
hold.settle_usage("gpt-4o", &Usage::new(18_000, 700));

// Close a scope to give the unused part back. Dropping it does the same,
// including when the sub-agent panics or its task is cancelled.
let receipt = research.settle();
println!("{} spent ${:.4}, ${:.4} back", receipt.scope_id, receipt.spent_usd, receipt.released_usd);
```

What each step does:

1. `child` takes the whole slice from the parent with a compare-and-swap.
   Siblings racing for the last dollar cannot all be admitted; the loser
   gets `BudgetError::AgentExhausted` before it spends anything.
2. Every amount a child books is booked on each ancestor at the same moment,
   out of the child's hold. A parent's `spent_usd()` is current while its
   children run, and its audit log has every child call under the child's
   name.
3. `settle()` (or drop) closes the child. A closed scope refuses new
   reservations, charges and children with `BudgetError::ScopeClosed`, for
   itself and every scope below it. Calls already in flight still settle;
   the part of the hold they need stays reserved until they do, and the rest
   goes back to the parent immediately.

### The invariant

For every scope, under any interleaving of threads or tasks: open child
holds plus everything booked never exceed its limit through admission.
`tests/scopes.rs` checks it with 64 children (and a grandchild each) running
at once while a monitor thread samples the parent's `utilization()`, and
`tests/scope_props.rs` checks the books after random sequences of 1 to 80
operations (children, reservations, settlements, overruns, forced charges,
closes) on trees up to depth 4. Three race tests (since 0.8.1) close a scope
while another thread, holding the scope's `Arc` the way an integration does,
reserves, charges or opens a grandchild on it, then admit a sibling for
whatever the parent says is free, 5,000 to 20,000 times each.

The one way past a limit is a settlement larger than its reservation: a
call that cost more than estimated, or `charge_usd_force`. That money is
already spent, so it is booked in full on every level and `over_cap` /
`ScopeReceipt::over_hold` report it. Reserve the worst case (prompt plus
`max_tokens`) to rule it out.

### Own cost and child cost

`own_cost_usd()` is what a scope's own calls spent; `child_cost_usd()` is
what its descendants spent; they add up to `spent_usd()`. The split matches
the A2A budget extension draft
([a2aproject/A2A#2121](https://github.com/a2aproject/A2A/issues/2121)), where
`incurred.cost` is an agent's own spend and `aggregatedChildCost` the
subtree below it, and the OpenTelemetry GenAI cost proposal, where an
operation's cost must not include nested operations.

### Depth

`AgentBudgetBuilder::max_depth(n)` caps how deep a tree can grow (default
8; the root is depth 0). A deeper `child` fails with
`BudgetError::DepthExceeded { scope_id, depth, max_depth }`.

### Kill switches

`kill()` on any scope stops it and every scope below it. Siblings and
ancestors keep running.

## On tokio

`ScopeGuard` is `Send + Sync + 'static`. Move it into the task:

```rust
let mut tasks = Vec::new();
for i in 0..4 {
    let scope = parent.child(format!("sub-{i}"), Limit::Usd(0.25))?;
    tasks.push(tokio::spawn(async move {
        run_sub_agent(&scope).await; // &ScopeGuard derefs to &AgentBudget
        scope.settle()
    }));
}
```

Integrations that take an `Arc<AgentBudget>` (such as `rig::BudgetHook`)
get one from `scope.budget().clone()`; see `examples/rig_scopes.rs`.

## Models with no price

A sub-agent may call a model your price table does not know. Since 0.7 the
reason is typed (`PriceTable::lookup` returns a `PriceStatus`: `Exact`,
`Unpriced`, `Missing` or `NoCatalog`, mirroring agentgateway's
`CostLookupStatus`), and an `UnpricedPolicy` picks `Allow`, `Reserve(usd)` or
`Deny` for each status. Child scopes inherit their parent's policy and
price table.

```rust
use llm_budget::{PriceAction, UnpricedPolicy};

let policy = UnpricedPolicy::new(
    PriceAction::Reserve(0.05), // the catalog lists it without a price: hold $0.05
    PriceAction::Deny,          // no source knows the id (often a typo): refuse
    PriceAction::Deny,          // no price data at all: refuse
);
let parent = AgentBudget::builder("orchestrator", 1.00).unpriced_policy(policy).build()?;
```

Without a policy nothing changes from 0.6: an unpriced model is refused
with `BudgetError::UnknownModel`.

## Across process boundaries (0.8)

A child scope works inside one process. When the sub-agent is a remote A2A
agent or an MCP tool server, `llm_budget::context` carries the slice to it
and the cost back:

```rust
use llm_budget::context::{a2a, mcp, BudgetContext, IncurredBooker};

// Caller: reserve a slice and send it with the A2A message.
let slice = parent.child("remote-researcher", Limit::Usd(0.30))?;
let metadata = a2a::to_metadata(&BudgetContext::outbound(&slice))?;

// Callee: a budget of exactly what it was given.
let ctx = a2a::from_metadata(&metadata)?.expect("budget extension present");
let agent = AgentBudget::from_context("researcher", &ctx)?;

// Callee to an MCP tool: the same budget as W3C baggage in params._meta.
let mut meta = serde_json::Map::new();
mcp::inject(&mut meta, &BudgetContext::outbound(&tool_slice));

// Callee reports back in the task metadata; caller books it once.
let report = a2a::to_metadata(&BudgetContext::reporting(&agent))?;
let mut booker = IncurredBooker::new();
booker.book(&slice, a2a::from_metadata(&report)?.and_then(|c| c.incurred).as_ref());
```

The wire format, under the extension URI
`https://a2a-protocol.org/extensions/budget/v1`:

```json
{
  "budget": {"cost": {"amount": "0.3", "currency": "USD"}, "tokens": {"limit": 5000}, "deadline": "2026-12-01T00:00:00Z"},
  "reservation": {"id": "rsv_a2dcda7352c5fe3c882a05e2b66a162a", "amount": "0.3", "currency": "USD"},
  "incurred": {
    "cost": {"amount": "0.04", "currency": "USD", "source": "local"},
    "aggregatedChildCost": {"amount": "0.02", "currency": "USD"},
    "tokens": {"input": 1200, "output": 340},
    "delegations": 1
  }
}
```

- `budget` and `incurred` use the field names of the A2A draft
  ([a2aproject/A2A#2121](https://github.com/a2aproject/A2A/issues/2121)).
  The draft is not final: its author has said amounts will be decimal
  strings (done here), that `aggregatedChildCost` is a provisional name, and
  that token counts gain cache fields. This crate follows the draft in minor
  releases and keeps reading what it wrote before.
- `reservation` is this crate's addition: the slice the parent reserved,
  with an id unique to that reservation. The id is never shared between
  siblings, so a callee cannot tell how many siblings it has.
- In baggage the outbound half becomes `llm.budget.cost.amount`,
  `llm.budget.cost.currency`, `llm.budget.tokens.limit`,
  `llm.budget.deadline`, `llm.budget.reservation.id`, `.amount`,
  `.currency` and `.expires`. MCP reserves the `baggage` key of `_meta` for
  W3C baggage (SEP-414), so no MCP change is needed.
- Absent is not zero: anything a reporter did not measure is left out.
- `incurred` is cumulative in a stream; `IncurredBooker` books only growth.

The Python package [agent-budget](https://github.com/Mattbusel/agent-budget)
implements the same contexts, the same reserve/settle scopes, and an
a2a-python plugin that does all of the above on both sides of an A2A call.
Both libraries' tests read the fixtures in `tests/fixtures/budget_context/`.
Run the whole flow offline with `cargo run --example budget_context`.
