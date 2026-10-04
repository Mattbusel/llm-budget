//! Respect a provider's tokens-per-minute limit and a dollar budget at the same
//! time, and log every decision with `tracing`.
//!
//! Run with `cargo run --example rate_limit --features ratelimit,tracing`.

use llm_budget::ratelimit::TokenRateLimiter;
use llm_budget::{AgentBudget, BudgetError, ModelTier};

#[tokio::main]
async fn main() -> Result<(), BudgetError> {
    tracing_subscriber::fmt().with_max_level(tracing::Level::DEBUG).init();

    // 30,000 tokens and 500 requests per minute, and $0.25 to spend.
    let limiter = TokenRateLimiter::per_minute(30_000, Some(500))?;
    let agent = AgentBudget::new("summarizer", 0.25, ModelTier::Standard, 0.2)?;

    for call in 1..=4 {
        let (input, output) = (9_000, 1_000);
        match limiter.check(input + output) {
            Ok(()) => {
                let r = agent.charge_model("gpt-4o", input, output, None, None).await?;
                println!("call {call}: sent, cost ${:.4}, ${:.4} left", r.cost_usd, agent.remaining_usd());
            }
            Err(BudgetError::RateLimited { retry_after_ms }) => {
                println!("call {call}: held back, retry in {retry_after_ms} ms");
            }
            Err(e) => return Err(e),
        }
    }
    Ok(())
}
