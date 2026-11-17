//! A hard USD cap on an OpenAI-compatible HTTP client, as a tower layer.
//!
//! ```text
//! cargo run --example tower_budget --features tower
//! ```
//!
//! The "client" here is a stand-in service that answers like the OpenAI Chat Completions API,
//! so the example runs offline. With a real client, wrap whatever tower `Service` sends the
//! request (a hyper client, `tower_http` stack, or a small adapter around reqwest that buffers
//! the body into `Bytes`): the layer only needs `http::Request<Bytes>` in and
//! `http::Response<Bytes>` out.

use std::sync::Arc;

use bytes::Bytes;
use llm_budget::tower_layer::{BudgetLayer, BudgetLayerError};
use llm_budget::AgentBudget;
use tower::{service_fn, Layer, Service, ServiceExt};

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // $0.02 for this job, priced from llm-prices (read from the providers' pricing pages).
    let budget = Arc::new(AgentBudget::builder("nightly-summaries", 0.02).build()?);

    let api = service_fn(|req: http::Request<Bytes>| async move {
        // Pretend the provider used 3,000 prompt tokens and wrote 600.
        let model = serde_json::from_slice::<serde_json::Value>(req.body())
            .ok()
            .and_then(|v| v.get("model").and_then(|m| m.as_str()).map(str::to_string))
            .unwrap_or_default();
        let body = serde_json::json!({
            "model": model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "..."}}],
            "usage": {"prompt_tokens": 3_000, "completion_tokens": 600}
        });
        Ok::<_, std::convert::Infallible>(http::Response::new(Bytes::from(body.to_string())))
    });
    let mut client = BudgetLayer::openai_http(Arc::clone(&budget)).layer(api);

    for i in 1..=20 {
        let body = serde_json::json!({
            "model": "gpt-4.1-mini",
            "max_tokens": 800,
            "messages": [{"role": "user", "content": format!("Summarize report #{i} in five bullet points.")}]
        });
        let req = http::Request::post("https://api.openai.com/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Bytes::from(body.to_string()))?;
        match client.ready().await?.call(req).await {
            Ok(_) => println!("call {i:>2}: sent, spent ${:.6} of $0.02", budget.spent_usd()),
            Err(BudgetLayerError::Refused(e)) => {
                println!("call {i:>2}: refused before sending ({e})");
                break;
            }
            Err(BudgetLayerError::Inner(e)) => match e {},
        }
    }
    Ok(())
}
