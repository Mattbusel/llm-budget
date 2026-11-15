//! A dollar cap around async-openai chat completions, plain and streamed.
//!
//! Runs offline: the responses below are the JSON OpenAI returns, parsed into
//! async-openai's own types. With a real client, replace them with
//! `client.chat().create(request).await?` and
//! `client.chat().create_stream(request).await?`.
//!
//! Run with `cargo run --example async_openai_budget --features async-openai`.

use async_openai::types::chat::{
    CreateChatCompletionRequest, CreateChatCompletionResponse, CreateChatCompletionStreamResponse,
};
use llm_budget::{openai, AgentBudget};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let budget = AgentBudget::builder("support-bot", 0.02).build()?;

    // A plain call.
    let request: CreateChatCompletionRequest = serde_json::from_str(
        r#"{"model":"gpt-4o","max_completion_tokens":400,
            "messages":[{"role":"system","content":"You answer billing questions."},
                        {"role":"user","content":"Why was I charged twice this month?"}]}"#,
    )?;
    let hold = openai::reserve_request(&budget, &request)?;
    println!("held ${:.5} for {}", hold.estimate_usd(), request.model);
    let response: CreateChatCompletionResponse = serde_json::from_str(
        r#"{"id":"chatcmpl-1","object":"chat.completion","created":1,"model":"gpt-4o-2024-08-06",
            "choices":[],"usage":{"prompt_tokens":1450,"completion_tokens":210,"total_tokens":1660,
            "prompt_tokens_details":{"cached_tokens":1024}}}"#,
    )?;
    let outcome = openai::settle_response(hold, &response);
    println!("booked ${:.5}, ${:.5} left", outcome.recorded_usd, budget.remaining_usd());

    // A streamed call: ask for the usage chunk, collect it, settle at the end.
    let mut request: CreateChatCompletionRequest = serde_json::from_str(
        r#"{"model":"gpt-4o-mini","stream":true,"max_completion_tokens":800,
            "messages":[{"role":"user","content":"Draft a refund email."}]}"#,
    )?;
    openai::include_usage(&mut request);
    let hold = openai::reserve_request(&budget, &request)?;
    let mut usage = openai::StreamUsage::new();
    for chunk in [
        r#"{"id":"c2","object":"chat.completion.chunk","created":1,"model":"gpt-4o-mini","choices":[],"usage":null}"#,
        r#"{"id":"c2","object":"chat.completion.chunk","created":1,"model":"gpt-4o-mini","choices":[],
            "usage":{"prompt_tokens":620,"completion_tokens":540,"total_tokens":1160}}"#,
    ] {
        let chunk: CreateChatCompletionStreamResponse = serde_json::from_str(chunk)?;
        usage.observe(&chunk);
    }
    let outcome = usage.settle(hold);
    println!("stream booked ${:.5}, ${:.5} left", outcome.recorded_usd, budget.remaining_usd());

    // A request too big for what is left is refused before it is sent.
    let huge: CreateChatCompletionRequest =
        serde_json::from_str(r#"{"model":"gpt-4o","max_completion_tokens":100000,"messages":[]}"#)?;
    match openai::reserve_request(&budget, &huge) {
        Ok(_) => println!("unexpected: the big request fit"),
        Err(e) => println!("big request not sent: {e}"),
    }
    Ok(())
}
