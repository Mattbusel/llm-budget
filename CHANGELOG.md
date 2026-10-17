# Changelog

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
