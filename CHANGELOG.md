# Changelog

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
