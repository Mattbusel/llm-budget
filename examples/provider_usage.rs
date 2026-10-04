//! Charge from the `usage` block each provider returns, with prompt-cache
//! discounts, whatever HTTP client or SDK you use. The response bodies below
//! are the shapes OpenAI, Anthropic and Gemini actually return.
//!
//! Run with `cargo run --example provider_usage`.

use llm_budget::{AgentBudget, BudgetError, ModelTier, Usage};

#[tokio::main]
async fn main() -> Result<(), BudgetError> {
    let agent = AgentBudget::new("support-bot", 1.0, ModelTier::Standard, 0.2)?;

    let calls = [
        (
            "gpt-4o",
            r#"{"id":"chatcmpl-1","object":"chat.completion","choices":[],
                "usage":{"prompt_tokens":12000,"completion_tokens":400,"total_tokens":12400,
                         "prompt_tokens_details":{"cached_tokens":10240}}}"#,
        ),
        (
            "claude-sonnet-4-5",
            r#"{"id":"msg_1","type":"message","content":[],
                "usage":{"input_tokens":300,"output_tokens":250,
                         "cache_creation_input_tokens":0,"cache_read_input_tokens":11000}}"#,
        ),
        (
            "gemini-2.5-flash",
            r#"{"candidates":[],"usageMetadata":{"promptTokenCount":9000,"candidatesTokenCount":300,
                "thoughtsTokenCount":700,"totalTokenCount":10000}}"#,
        ),
    ];

    for (model, body) in calls {
        let usage = Usage::from_json(body)?;
        let without_cache = llm_budget::PriceTable::builtin()
            .require(model)?
            .cost(usage.input_tokens, usage.output_tokens);
        let r = agent.charge_usage(model, &usage, None, None).await?;
        println!(
            "{model:<18} {:>6} in ({:>6} cached) {:>4} out: ${:.5} (${:.5} without the cache discount)",
            usage.input_tokens, usage.cached_input_tokens, usage.output_tokens, r.cost_usd, without_cache
        );
    }
    println!("total ${:.5}, ${:.5} left", agent.spent_usd(), agent.remaining_usd());
    Ok(())
}
