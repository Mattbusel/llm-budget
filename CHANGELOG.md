# Changelog

## [0.8.2] - 2026-11-17

For projects deciding whether to depend on this crate. No API changes.

### Added
- `docs/INTEGRATING.md`: the integration surface in one page. License, MSRV, no unsafe, no panics, no network or globals, what `default-features = false` pulls in, every feature with its dependencies, how to make llm-budget an optional dependency of your own crate, the five-line reserve/settle integration, which entry point fits which stack (any client, tower, tracing, metrics, rig, genai, async-openai, sub-agents, A2A/MCP), price overrides, and the stability policy.
- `tests/api_surface.rs`: compile-time checks that every shared type (`AgentBudget`, `Reservation`, `ScopeGuard`, `FleetAllocator`, `PriceTable`, `BudgetError`, `BudgetContext`, receipts and records) is `Send + Sync + 'static`, that `BudgetError` boxes into `Box<dyn Error + Send + Sync>`, and that the five-line integration from the docs runs without default features.
- CI job `semver`: cargo-semver-checks compares the public API with the latest crates.io release.
- `#![forbid(unsafe_code)]` (there was no unsafe code; now it cannot be added by accident).

### Changed
- thiserror 1 to 2 and dashmap 5 to 6. Neither appears in the public API. Projects already on thiserror 2 (rig 0.46, most of the current ecosystem) no longer compile two copies of it: with every feature on, the tree had both thiserror 1.0.69 and 2.0.21; now it has one.
- `docs/DETAILS.md`: the features table lists `metrics-layer` and `tower`, and `rig` covers rig-agent 0.44 to 0.46.

## [0.8.1] - 2026-11-17

### Fixed
- A child scope closed while another thread was taking a reservation on it (an integration such as `rig::BudgetHook` holding `scope.budget()` while the orchestrator settles or drops the guard on a timeout) could admit that reservation after the close had already refunded the whole hold, so the parent's books went past its cap once a sibling took the refunded room. `reserve`, `child` and envelope charges now take the hold and then check the scope again (store-then-load, `SeqCst`, the same pattern as `shrink_limit`): either the close keeps the hold, or the reservation sees the close, gives the hold back and returns `ScopeClosed`. `BudgetEnvelope::charge` now books through a reservation, so a charge racing a close is covered the same way. A review reproduced it as $1.50 committed on a $1.00 root; the new race test fails on 0.8.0 and passes on 0.8.1.
- Docs and the 0.8.0 notes linked agent-budget's PyPI page before its first upload; they link the repository.

### Added
- `IncurredBooker::capped_at(amount)` and `IncurredBooker::excess()`: book at most the reservation a callee was given and keep what it reported above that, so one buggy callee cannot push every ancestor past its cap.
- Tests: `tests/scopes.rs` races a close against `reserve`, `charge_usd` and `child` on the same scope; `tests/context_fixtures.rs` covers the capped booker and a saturating report. Shared fixtures gain JSON-number amounts too large to hold (`1e300`, `18446744074`), counts given under both `cacheRead` and `cache_read`, and an exact large whole-number amount.

## [0.8.0] - 2026-11-17

Budgets that cross process boundaries. A `BudgetContext` in the shape of the A2A budget extension draft travels in A2A metadata, W3C baggage and MCP `_meta`, so a remote agent or tool server gets exactly the slice its caller reserved and reports back what it spent. The Python package [agent-budget](https://github.com/Mattbusel/agent-budget) speaks the same format; both test suites read the same golden fixtures.

### Added
- Module `context`:
  - `BudgetContext { budget, incurred, reservation }` with the draft's field names (a2aproject/A2A#2121): `budget{cost, tokens{limit}, deadline}`, `incurred{cost{amount, currency, source}, aggregatedChildCost, tokens{input, output, cacheRead, cacheWrite, total}, delegations}`, and an optional `reservation{id, amount, currency, expires}` for the slice a parent reserved. Money is a decimal string on the wire (`Amount`, exact to the nano-unit); JSON numbers and whole floats (protobuf `Struct`) are read too. Absent fields stay absent, never zero. `from_json` rejects malformed known fields and ignores unknown ones.
  - `BudgetContext::outbound(&scope)` (what a child scope has left plus its reservation, with a fresh `rsv_` id per reservation), `remaining(&budget)`, `reporting(&budget)` and `Incurred::of(&budget, source)` (own cost, `aggregatedChildCost` from child scopes, `delegations`), `with_deadline`, `with_token_limit`, `budget_usd`. `AgentBudget::from_context(id, &ctx)` builds a callee budget capped at what it was given. `AgentBudget::delegations()`.
  - `IncurredBooker` books a callee's cumulative `incurred` report on the caller's scope once, however many times it arrives.
  - Carriers: `a2a::{EXTENSION_URI, to_metadata, insert, from_metadata}`; `baggage::{encode, decode, merge, members}` with `llm.budget.*` keys, W3C percent-encoding and the 8,192-byte limit; `mcp::{inject, extract}` for `params._meta.baggage` (reserved for W3C baggage by MCP SEP-414).
- Golden fixtures in `tests/fixtures/budget_context/` (round trips, normalization, inputs both libraries must reject), shared with agent-budget; `tests/context_fixtures.rs` runs them and fuzzes the untrusted readers (proptest: `baggage::decode` and `from_json` never panic, every valid outbound context round-trips through JSON and baggage, amounts print and parse back exactly).
- `examples/budget_context.rs`: an orchestrator, a remote A2A agent and an MCP tool server passing one budget down and the cost back up, offline. CI runs it.

## [0.7.0] - 2026-11-17

Budget scopes for agents that delegate: a parent reserves a slice for each sub-agent, the children of a delegation together can never be admitted for more than the parent has left, and unused slices are refunded. Plus a typed policy for models with no price, and the proposed OpenTelemetry cost attribute names in `CostLayer`. Guide: `docs/budget-scopes.md`.

### Added
- Child scopes (module `budget::scope`):
  - `AgentBudget::child(id, Limit::Usd(x) | Limit::Fraction(f))` reserves the slice from the parent with a compare-and-swap and returns a `ScopeGuard`, which derefs to a full `AgentBudget` (`reserve_for`, `record`, `charge_*`, `kill` and `child` for grandchildren). The child inherits the parent's price table, unpriced policy, audit-log setting, tier and cascade threshold. `BudgetScope` is an alias of `AgentBudget` naming the role.
  - Everything a child books is booked on every ancestor at the same moment, out of its hold, with the child's name in each ancestor's audit log. `own_cost_usd()` and `child_cost_usd()` split a scope's spend into its own calls and its descendants' (the A2A budget draft's `cost` / `aggregatedChildCost`).
  - `ScopeGuard::settle()` (or drop, including on a panic or a cancelled task) closes the scope and refunds the unused hold at once; calls in flight keep their part until they settle. It returns a `ScopeReceipt` (`held_usd`, `spent_usd`, `own_cost_usd`, `child_cost_usd`, `released_usd`, `in_flight_usd`, `over_hold`). A closed scope, and every scope below it, refuses new work with the new `BudgetError::ScopeClosed`. Killing a scope stops its whole subtree.
  - `AgentBudgetBuilder::max_depth(n)` (default `DEFAULT_MAX_DEPTH` = 8) and the new `BudgetError::DepthExceeded { scope_id, depth, max_depth }`. `AgentBudget::depth`, `max_depth`, `parent_id`, `is_closed`, `utilization`. `ScopeGuard` is `Send + Sync + 'static`; `ScopeGuard::budget()` gives the child as an `Arc<AgentBudget>` for integrations such as `rig::BudgetHook`.
- Typed unpriced-model policy:
  - `PriceTable::lookup(model, &usage)` returns a `PriceLookup` with a `PriceStatus`: `Exact`, `Unpriced` (known to the catalog without a usable price), `Missing` (no source knows the id) or `NoCatalog` (no price data at all). The names and their serde spellings (`exact`, `unpriced`, `missing`, `noCatalog`) mirror agentgateway's `CostLookupStatus`. `PriceTable::with_unpriced(model)` declares a model as known but unpriced.
  - `UnpricedPolicy { unpriced, missing, no_catalog }`, one `PriceAction` (`Allow`, `Reserve(usd)`, `Deny`) per status, set with `AgentBudgetBuilder::unpriced_policy`. `reserve_for`, `record`, and every integration built on them (tower `BudgetLayer`, genai, async-openai) follow it; `Deny` returns the new `BudgetError::Unpriced { model, status }`. `Reservation::price_status()`, `AgentBudget::price_status()` and `unpriced_policy()`. Without a policy the behavior is unchanged (`UnknownModel`).
- Module `semconv`: the `gen_ai.usage.cost.amount` / `.currency` / `.source` names and `CostSource` (`provider` / `local`).
- `examples/fanout.rs` (four parallel sub-agents under one cap, compared with passing the remainder down) and `examples/rig_scopes.rs` (a rig orchestrator with two sub-agents in child scopes, offline). CI runs both.
- Tests: `tests/scopes.rs` (the two-delegate overshoot of a propagated remainder and the same delegates under child scopes, 64 parallel children with grandchildren under a utilization monitor, a panicking sub-agent, a cancelled tokio task, fraction slices raced from 8 threads), `tests/scope_props.rs` (proptest: random trees of scopes, holds, settlements, overruns and closes keep every scope's books exact), `tests/unpriced_policy.rs` (every `PriceStatus` under every `PriceAction`).

### Changed
- `tracing_layer::CostLayer` cost events use the attribute names proposed in open-telemetry/semantic-conventions-genai#443: `gen_ai.usage.cost.amount`, `gen_ai.usage.cost.currency` (`USD`) and `gen_ai.usage.cost.source` (`local` when the layer priced the call, the span's own source for a reported cost). The 0.5/0.6 field `gen_ai.usage.cost` is no longer emitted; it is still read as a reported cost on spans, next to `gen_ai.usage.cost.amount`. A reported amount whose `gen_ai.usage.cost.currency` is not `USD` is ignored and the call is priced from the table. `SpanCost::source` added.
- A test that expected `openai/gpt-4o` to resolve to the bare `gpt-4o` row now uses an unlisted prefix: since llm-prices 1.6, `openai/gpt-4o` is an OpenRouter row of its own.

## [0.6.0] - 2026-11-16

A hard USD cap as a tower `Layer`, for any client or proxy built on tower services.

### Added
- Feature `tower`, module `tower_layer`:
  - `BudgetLayer` / `BudgetService`: before each call, an estimator reads the request (model plus expected usage) and takes a `reserve_for` hold. A call that does not fit under the cap, or after the kill switch, fails with `BudgetLayerError::Refused` and the inner service is never called. On success the provider's reported usage is booked (`settle_usage`); with no usage in the response the estimate is booked; when the inner service fails the hold is released and nothing is booked.
  - `BudgetLayer::openai_http(budget)`: estimator and usage reader for OpenAI-compatible `http::Request<Bytes>` / `http::Response<Bytes>` calls (Chat Completions and Responses API bodies; Anthropic and Gemini usage shapes are read too). `estimate_openai_body` and `read_openai_body` are public for your own layers. `BudgetLayer::new(budget, estimate, read_usage)` takes closures for any request and response types.
- `examples/tower_budget.rs`: an OpenAI-compatible client capped at $0.02, offline; call 10 is refused before it is sent. CI runs it, and the MSRV job tests the feature on Rust 1.85.

## [0.5.1] - 2026-11-15

Price the token counters a framework already emits through the metrics facade. A new `metrics-layer` feature adds `metrics_layer::CostLayer`, the metrics twin of 0.5.0's tracing layer, with a preset for swiftide.

### Added
- Feature `metrics-layer`, module `metrics_layer`:
  - `CostLayer`, a `metrics_util::layers::Layer`: wrap any recorder (Prometheus, StatsD, OpenTelemetry, a test recorder) and every increment of a configured token counter is priced and added to a cost gauge with the counter's labels (USD; a gauge because metrics counters hold whole numbers, and it is only ever incremented). The original counters pass through unchanged. Tokens on a model no source can price go to `llm_budget_unpriced_tokens_total` instead, never guessed.
  - `CostLayer::swiftide()`: preset for swiftide's `swiftide.usage.prompt_tokens` / `swiftide.usage.completion_tokens` (label `model`), recording `swiftide.usage.cost_usd`. `input_tokens`, `output_tokens`, `model_label` and `cost_metric` for any other emitter; `price_table` for your own prices.
  - `CostLayer::budget(Arc<AgentBudget>)` books every priced increment (even past the cap, since it was paid for) and, once over the cap, trips the kill switch so `reserve_for` and `charge_*` refuse the next call. `kill_on_cap(false)` books without killing.
- `examples/metrics_cost.rs`: swiftide-style counters priced and capped, offline. CI runs it.

## [0.5.0] - 2026-11-15

Price LLM calls from their traces. A new `tracing-layer` feature adds `CostLayer`, a tracing-subscriber layer that turns the OpenTelemetry GenAI usage fields rig (and other instrumentations) already record into dollar costs, with an optional cap. Prices come from llm-prices 1.3.

### Added
- Feature `tracing-layer`, module `tracing_layer`:
  - `CostLayer`: watches spans that declare `gen_ai.usage.input_tokens` / `gen_ai.usage.output_tokens` (or the older `prompt_tokens` / `completion_tokens`). When one closes it prices the call (model from `gen_ai.response.model`, then `gen_ai.request.model`; cache reads and writes from `gen_ai.usage.cache_read.input_tokens` and `gen_ai.usage.cache_creation.input_tokens`) and emits an `INFO` event on target `llm_budget::cost`, inside the span, with `gen_ai.usage.cost` in USD next to the model and token counts. A span that records its own `gen_ai.usage.cost` keeps that figure. A span whose descendants were already priced (rig's `invoke_agent` sum over its chat turns) is not counted again. Models no source can price are counted, never guessed.
  - `CostLayer::budget(Arc<AgentBudget>)`: books every priced call (even past the cap, since it was paid for) and, once spend is over the cap, kills the budget and emits a `WARN` event, so `reserve_for`, `charge_*` and `rig::BudgetHook` on that budget refuse the next call. `kill_on_cap(false)` books without killing.
  - `CostLayer::price_table`, `emit_events`, `on_cost(|&SpanCost| ..)`, and `CostTracker` (`total_usd`, `priced_calls`, `unpriced_calls`, `over_cap`) from `CostLayer::tracker()`.
- `examples/rig_tracing_cost.rs`: a plain rig agent (no hook) priced from its spans, with a cap, offline against rig's mock model. CI runs it.
- `tests/tracing_layer.rs`: rig-shaped chat spans, cache-read pricing, no double count of the agent sum, unknown models, custom tables, reported costs, older field names, budget booking and the kill at the cap.

### Changed
- llm-prices requirement raised from 1.1 to 1.3 (Mistral, xAI and more providers, dated-snapshot and provider-prefixed id resolution). Dated ids such as `gpt-4o-2099-01-01` now resolve to their base model's price instead of falling back to the reserved model; two tests that relied on the old miss now use an id no source lists.
- MSRV job also tests `tracing-layer` on Rust 1.85.

## [0.4.0] - 2026-11-14

Report spend from any client or framework, integrations for genai and async-openai, prices from the llm-prices crate, and a core with no async runtime. Pre-1.0 minor bump: the API is additive, but tokio moved behind a default feature and some built-in prices changed (see Changed).

### Added
- Framework-agnostic reporting, the core pattern for any client:
  - `AgentBudget::reserve_for(model, &estimate)` holds the estimated cost of a call (priced from the agent's table) and refuses when it does not fit, so the call is never made.
  - `Reservation::settle_usage(model, &usage)` / `settle_usage_with(.., ChargeContext)` book what the provider reported. If `model` has no price (a dated id echoed back) the reserved model is tried, then the reserved estimate is booked. Never fails; books past the cap and sets `over_cap`. `Reservation::model()`.
  - `AgentBudget::record(model, &usage)` / `record_with(.., ChargeContext)` book a finished call that had no reservation, even past the cap. `UnknownModel` books nothing.
  - `AgentBudget::cost_of(model, &usage)` prices a call without booking it.
  - `Usage::from_parts(uncached_input, cache_read, cache_write, output)`, `with_cache_read`, `with_cache_write`, `total_tokens`, `is_empty`, and `Add`, `AddAssign` and `Sum` for totalling turns. `usage::estimate_tokens(text)` and `usage::estimate_json_tokens(&value)` for sizing a reservation.
  - `ChargeContext::usage(&usage)` fills the token counts.
- Feature `genai`: `llm_budget::genai::exec_chat(&client, &budget, model, request, options)` reserves (prompt estimated from the `ChatRequest`, output from `max_tokens`), calls genai and settles with genai's normalised usage; a genai error releases the hold (`genai::Error::{Budget, Genai}`). `exec_chat_stream` turns on `capture_usage` and returns the stream with its reservation; `settle_stream_end` settles from the `End` event. Also `settle_response`, `estimate_request` and `From<&genai::chat::Usage> for Usage` (cache reads and writes, including per-TTL writes). A genai `namespace::` prefix is ignored for pricing.
- Feature `async-openai`, new `openai` module: `reserve_request(&budget, &CreateChatCompletionRequest)`, `settle_response(reservation, &CreateChatCompletionResponse)`, `record_response`, `estimate_request`, `include_usage(&mut request)` (sets `stream_options.include_usage`) and `StreamUsage`, which collects the final usage chunk of a stream and settles with it (or with the estimate when the stream ended without one).
- Prices from the [llm-prices](https://crates.io/crates/llm-prices) crate (1.1, no dependencies): dated, source-cited prices for Anthropic, OpenAI, Gemini and DeepSeek now win over the LiteLLM snapshot for every model they list, and its id lookup resolves Bedrock and Vertex ids (`us.anthropic.claude-sonnet-4-6-v1:0`, `claude-opus-4-1@20250805`). The LiteLLM snapshot stays for the other providers (Mistral, xAI, older ids) and for context sizes. `pricing::LLM_PRICES_DATE`.
- `PriceTable::cost_usage(model, &usage)` and `PriceTable::cost(model, in, out)`: price through the lookup order hook, inserted prices, llm-prices (with long-context tiers: a 300k-token prompt on gpt-5.4 bills at $5/M, not $2.50/M), then the table's rows. OpenAI rows that only list 30m/12h cache-write prices are priced at the shortest lifetime.
- Override hooks: `PriceTable::with_hook(|model, usage| Option<f64>)` (consulted first), `with_price`, `with_llm_prices(bool)`, `from_llm_prices()`, `contains(model)`; `ModelPrice::new(..)` and `with_cache(..)`; `AgentBudgetBuilder::price_table(table)` gives an agent (or, through an `Arc`, several) its own table, used by `charge_model`, `charge_usage`, `record`, `reserve_for` and `settle_usage`; `AgentBudget::price_table()`.
- Feature `store` (default): `BudgetStore` and `persistence`. Everything else needs no async runtime; `cargo test --no-default-features` builds without tokio.
- Examples: `record_usage` (the core pattern with your own prices, plain `fn main`, no runtime), `async_openai_budget` (requests and streams, offline), `genai_budget` (an agent loop and a stream; needs an API key).
- Tests: `tests/record_usage.rs`; `tests/genai_mock.rs` runs a real genai `Client` over HTTP against a local mock server (reported usage booked, refused call never sent, provider error releases the hold, unknown model); unit tests for the openai module, genai usage mapping, llm-prices layering, long-context and 30m cache-write pricing, overrides and hooks.
- CI runs the new examples and `cargo test --no-default-features`.

### Changed
- tokio is optional (feature `store`, on by default). With `default-features = false`, `BudgetStore` and `persistence` are gone; add `features = ["store"]` to keep them.
- Built-in prices: where llm-prices lists a model, its numbers replace the LiteLLM snapshot's (for example `claude-sonnet-5-5` cache reads $0.10/M instead of $0.20/M), and a row's provider and context size stay LiteLLM's. `PriceTable::len()` is larger (llm-prices rows and aliases the snapshot lacked).
- `PriceTable::get` also resolves ids through llm-prices on the built-in table. `PriceTable` implements `Debug` by hand (counts, not every row).
- `charge_model` and `charge_usage` price from the agent's table (the built-in one by default) through `PriceTable::cost_usage`, so they pick up long-context tiers. The rig hook does too.
- `PriceTable::insert` marks the model as overridden, so the inserted price wins over llm-prices.

## [0.3.1] - 2026-11-14

### Fixed

- Review fixes on top of 0.3.0 (settlement and rig hook edge cases, tests).

## 0.3.0 (2026-11-14)

Settlement correctness, a bounded audit log, and the rig hook for rig-agent 0.44 to 0.46. Pre-1.0 minor bump because two defaults change behaviour (see Changed).

### Fixed
- Spend that had already happened could be thrown away. `charge_usd`, `charge_usage` and the rig hook book a call *after* the provider has answered, and when that charge would cross the cap it was refused and nothing was recorded, so `spent_usd` and the audit log came out lower than the provider's bill exactly when the cap was hit. Post-call settlement now always books (see Added), and the rig hook uses it.
- The audit log grew without bound: one record per charge, forever, and `audit_log()` cloned all of it. It is now a ring of the last 1,024 records by default.

### Added
- `AgentBudget::reserve(estimate_usd) -> Result<Reservation, BudgetError>`: holds headroom with a compare-and-swap before a call, so concurrent agents cannot all be admitted on the last dollar. `Reservation::settle(actual_usd)` / `settle_with(actual_usd, ChargeContext)` release the hold and book the real cost, even past the cap; dropping a `Reservation` (or `release()`) frees the hold and books nothing.
- `AgentBudget::charge_usd_force(cost_usd, ChargeContext) -> ChargeOutcome`: books spend that already happened, ignoring the cap and the kill switch, and logs it. `ChargeOutcome { recorded_usd, total_spent_usd, over_cap, downgraded_to }` says whether to stop. Also `BudgetEnvelope::charge_force` and `AtomicBudget::charge_force`.
- `ChargeContext` (task, session, model, token counts) for the audit record of a settlement.
- `AuditLog::{None, Ring(n), Unbounded}`, set with `AgentBudget::builder(..).audit_log(..)`, `BudgetEnvelope::with_audit_log` or `FleetAllocator::with_audit_log`. `drain_audit_log()` takes the records out (to ship them to your own storage) and `audit_records_dropped()` counts evicted ones.
- `AgentBudget::builder(agent_id, limit_usd)` with `tier` (default Standard), `cascade_threshold` (default 0.2) and `audit_log` (default `Ring(1024)`).
- `reserved_usd()` on `AtomicBudget`, `BudgetEnvelope` and `AgentBudget`; `AgentBudget::is_exhausted()`.
- Feature `rig`: `rig::BudgetHook`, an `AgentHook` for rig-agent 0.44 to 0.46 (the dependency is now `>=0.44, <0.47`; tested against 0.46). Before each model call it reserves the estimated prompt cost (about four characters per token, priced from the `PriceTable`, plus `reserve_output_tokens(n)` if set) and stops the run if that does not fit; after each turn it settles the reservation with the real cost (the provider's reported cost when rig has one, otherwise the usage priced with cache discounts) and only then, if the budget is over its cap, trips the kill switch so other runs sharing the budget stop before their next call (`keep_alive_on_refusal` turns that off) and stops the run if the turn asked for tool calls. A final answer that crosses the cap is returned to the caller, since it has been paid for, instead of being turned into an error. Also `From<&rig_agent::completion::Usage> for Usage`. Needs Rust 1.95 with rig-agent 0.44 and 1.97.1 with 0.46; the rest of the crate stays on 1.85. The module docs explain how this differs from rig's own `examples/run_budget.rs` (dollars across models, runs and agents, versus token and tool-call counts for one run).
- `examples/rig_budget_hook.rs`: two agents sharing a dollar cap, offline against rig's mock model.
- Property tests: the audit log sums to `spent_usd` under any mix of refused, forced, reserved-and-settled and reserved-and-dropped charges; 64 threads reserving and settling never overshoot the cap by more than the settle deltas, and never at all when no call costs more than its reservation.
- `tests/rig_hook.rs`: the over-cap final answer is returned; a turn stopped by a hook added after `BudgetHook` is still booked; and a pinned test of the hook-order limitation below.
- Crate docs: an authorize-then-settle guide in the `budget` module and a short map of the API at the end of the crate root.

### Changed
- `AgentBudget::new`, `BudgetEnvelope::new` and fleet agents keep the last 1,024 audit records instead of all of them. Use `AuditLog::Unbounded` for the old behaviour.
- `remaining_usd()`, `is_exhausted()` and `utilization()` count in-flight reservations as used. `spent_usd()` counts settled spend only. Without reservations nothing changes.
- `utilization()` can exceed 1.0 after a forced charge.
- The cascade and the audit log use `std::sync::Mutex` instead of tokio locks (no critical section is held across an `.await`). The async signatures are unchanged.
- CI (GitHub and GitLab) now runs `cargo test` on 1.85, plus rustdoc with `-D warnings` and the rig example on stable.
- `cargo test` builds on the MSRV (1.85) again: the rig mock-model tests moved to `tests/rig_hook.rs` behind the internal `__rig-test-utils` feature, and rig-agent is no longer an unconditional dev-dependency.

### Known limitations
- `rig::BudgetHook` must be added before any other hook. rig-agent dispatches `on_model_turn_finished` in `add_hook` order and stops at the first hook that does not return `Continue`, so a hook added ahead of `BudgetHook` that stops a turn keeps that paid turn off the books (one that retries it gets it booked at its prompt-only estimate). The rig module docs and `BudgetHook::new` say so.
- Reservations, their settlement, `charge_usd_force` and the rig hook book against the agent only. On an agent taken from `FleetAllocator::get_agent` they are not added to `FleetBudget::spent_usd`, which still counts only the `charge_agent*` methods; sum the agents' `spent_usd()` for the fleet's real total.
- `BudgetEnvelope::with_audit_log` changes the mode for every clone of the envelope (they share one log); set it before cloning.

## 0.2.2 (2026-10-22)

- README only. The source code, examples, benches and CHANGELOG in the published crate are identical to 0.2.1. This version was published from a working tree, so the `.crate` carries no VCS info and there is no tag; its README differs from every committed version. The repository's `Cargo.toml` was not bumped at the time (it still said 0.2.1 until 0.3.0).

## 0.2.1 (2026-10-17)

- `repository` now points at GitHub (GitLab stays as the mirror); crates.io category `finance` replaced by `development-tools`; added a homepage. README: CI and MSRV badges. No code changes.

## 0.2.0 (2026-10-03)

### Added
- Real per-model prices: `PriceTable` with 232 chat models from OpenAI, Anthropic, Google Gemini, Mistral, DeepSeek and xAI, generated from LiteLLM's maintained price list (MIT) by `scripts/update_prices.py`. Includes prompt-cache prices and context sizes. `PriceTable::from_litellm_json` loads a newer copy at runtime.
- `AgentBudget::charge_model`, `charge_model_with`, `charge_usd` and `FleetAllocator::charge_agent_model`, returning a `ChargeReceipt` with the cost of that one call.
- Feature `tokens`: `tokens::count_tokens`, `tokens::cost_of_text` and `AgentBudget::charge_model_text`, using tiktoken-rs. Exact for OpenAI models, marked as an estimate for other vendors.
- `Usage`: reads the usage block of OpenAI (Chat Completions and Responses), Anthropic and Gemini responses and normalises their different prompt-cache counting (OpenAI and Gemini report cached tokens as part of the prompt, Anthropic reports them on top). `AgentBudget::charge_usage`, `FleetAllocator::charge_agent_usage` and `ModelPrice::cost_usage` charge cache reads and writes at their own prices.
- Feature `ratelimit`: `TokenRateLimiter`, tokens-per-minute and requests-per-minute limits on top of governor, returning `BudgetError::RateLimited { retry_after_ms }`.
- Feature `tracing`: an event for every accepted or refused charge, downgrade and kill.
- Feature `metrics`: `llm_budget_charges_total`, `llm_budget_spent_usd`, `llm_budget_remaining_usd`, `llm_budget_downgrades_total` through the metrics facade.
- Feature `schemars`: JSON Schema for the serialized types.
- Feature `async-openai`: `From<&CompletionUsage> for Usage` (types only, no HTTP client).
- Property tests (proptest) for the money invariants: spend never passes the cap, equals the sum of accepted charges, reallocation conserves the fleet total, odd floats never panic.
- `benches/vs_alternatives.rs`: the charge path against 0.1.1, a `Mutex<u64>` and governor.
- Examples `provider_usage`, `persist_restore`, `rate_limit`; CONTRIBUTING.md and GitLab issue templates; an MSRV job in CI.
- `ChargeReceipt` and `RemovedAgent` derive serde.
- `FleetAllocator::remove_agent` (returns the unspent cap to the fleet) and `get_agent`.
- `BudgetEnvelope::kill_reason`, `AgentBudget::audit_log`, `AgentBudget::limit_usd`.
- `examples/quickstart.rs`, `examples/tokens.rs`; the README is now the crate docs and its quickstart is a doctest.

### Fixed
- Fleet spend was wrong: `charge_agent` added the agent's running total to the fleet counter after every charge, so three $3 charges showed as $18. It now adds the cost of each charge.
- `FleetAllocator::reallocate` checked the request and then did nothing. It now moves unspent cap from one agent to another, safely against concurrent charges.
- A charge that fitted could be refused when a larger charge on another thread crossed the limit at the same moment (add-then-roll-back). Charges now use a compare-and-swap loop.
- Small charges were rounded to whole micro-dollars, so a 1-token call to a cheap model counted as $0. Amounts are now kept in nano-dollars.
- `BudgetStore` joined the session id onto the folder path unchecked, so `"../x"` wrote outside the store folder. Ids are now limited to letters, digits, `-`, `_` and `.`. Saves are written to a temporary file and renamed, so a crash cannot leave half a file.
- The kill reason was thrown away; the error now carries it.
- Registering the same agent id twice replaced the agent (resetting its spend) and allocated its cap a second time. It is now an error.

### Changed
- Removed unused dependencies (rust_decimal, uuid, chrono, tracing, async-trait) and cut tokio down to the `sync` and `fs` features.
- `BudgetError` is `#[non_exhaustive]` and has two new variants: `UnknownModel` and `Tokenizer`.
- `SpendRecord` has a new `model` field.
- MSRV is 1.85.

## 0.1.1

- First published version.
