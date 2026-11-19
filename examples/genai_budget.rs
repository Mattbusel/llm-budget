//! A dollar cap on an agent loop built on genai, one `Client` for every
//! provider. Each turn is reserved before it is sent and settled with the
//! usage genai reports; the loop stops when the cap says so.
//!
//! Needs a real key, for example `OPENAI_API_KEY` (or `ANTHROPIC_API_KEY`
//! with `MODEL=claude-haiku-4-5`):
//!
//! `cargo run --example genai_budget --features genai`

use futures_util::StreamExt;
use genai::chat::{ChatMessage, ChatOptions, ChatRequest, ChatStreamEvent};
use llm_budget::{genai as budgeted, AgentBudget};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = std::env::var("MODEL").unwrap_or_else(|_| "gpt-4o-mini".to_string());
    let key_var = if model.starts_with("claude") { "ANTHROPIC_API_KEY" } else { "OPENAI_API_KEY" };
    if std::env::var(key_var).is_err() {
        println!("set {key_var} to run this example against {model}");
        return Ok(());
    }

    let client = genai::Client::new()?;
    let budget = AgentBudget::builder("explainer", 0.01).build()?; // one cent
    let opts = ChatOptions::default().with_max_tokens(200);
    let mut req = ChatRequest::default().with_system("Answer in two sentences.");

    for question in ["Why is the sky blue?", "Why is it red at sunset?", "What about on Mars?"] {
        req = req.append_message(ChatMessage::user(question));
        let (res, outcome) = match budgeted::exec_chat(&client, &budget, &model, req.clone(), Some(&opts)).await {
            Ok(done) => done,
            Err(budgeted::Error::Budget(e)) => {
                println!("not sent: {e}");
                break;
            }
            Err(e) => return Err(e.into()),
        };
        let answer = res.first_text().unwrap_or_default().to_string();
        println!(
            "Q: {question}\nA: {answer}\n   ${:.6} (total ${:.6})\n",
            outcome.recorded_usd, outcome.total_spent_usd
        );
        req = req.append_message(ChatMessage::assistant(answer));
        if outcome.over_cap {
            break;
        }
    }

    // Streaming: the End event carries the usage.
    let (mut res, hold) =
        budgeted::exec_chat_stream(&client, &budget, &model, ChatRequest::from_user("Count to five."), Some(&opts))
            .await?;
    let mut hold = Some(hold);
    while let Some(event) = res.stream.next().await {
        match event? {
            ChatStreamEvent::Chunk(c) => print!("{}", c.content),
            ChatStreamEvent::End(end) => {
                if let Some(hold) = hold.take() {
                    let outcome = budgeted::settle_stream_end(hold, &end);
                    println!("\n   stream ${:.6}", outcome.recorded_usd);
                }
            }
            _ => {}
        }
    }
    println!("spent ${:.6} of ${:.2}", budget.spent_usd(), budget.limit_usd());
    Ok(())
}
