//! `llm_budget::genai` against a real genai `Client` talking HTTP to a local
//! OpenAI-compatible mock server, so the reserve -> call -> settle path runs
//! end to end without a network or an API key.

#![allow(clippy::unwrap_used)]

use genai::adapter::AdapterKind;
use genai::chat::{ChatMessage, ChatOptions, ChatRequest};
use genai::resolver::{AuthData, Endpoint, ServiceTargetResolver};
use genai::{Client, ModelIden, ServiceTarget};
use llm_budget::{genai as budgeted, AgentBudget, BudgetError, PriceTable, Usage};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

/// Serve `status` + `body` to every request; returns the base URL.
async fn mock_server(status: u16, body: &'static str) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        loop {
            let Ok((mut sock, _)) = listener.accept().await else { return };
            tokio::spawn(async move {
                let mut buf = Vec::new();
                let mut chunk = [0u8; 4096];
                // read headers, then the body by content-length
                loop {
                    let n = sock.read(&mut chunk).await.unwrap_or(0);
                    if n == 0 {
                        return;
                    }
                    buf.extend_from_slice(&chunk[..n]);
                    let text = String::from_utf8_lossy(&buf).to_string();
                    if let Some(end) = text.find("\r\n\r\n") {
                        let len = text[..end]
                            .lines()
                            .find_map(|l| {
                                let (k, v) = l.split_once(':')?;
                                k.eq_ignore_ascii_case("content-length").then(|| v.trim().parse::<usize>().ok())?
                            })
                            .unwrap_or(0);
                        if buf.len() >= end + 4 + len {
                            break;
                        }
                    }
                }
                let reason = if status == 200 { "OK" } else { "Error" };
                let resp = format!(
                    "HTTP/1.1 {status} {reason}\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
                    body.len()
                );
                let _ = sock.write_all(resp.as_bytes()).await;
                let _ = sock.shutdown().await;
            });
        }
    });
    format!("http://{addr}/v1/")
}

fn client(base: String) -> Client {
    let resolver = ServiceTargetResolver::from_resolver_fn(move |t: ServiceTarget| -> Result<ServiceTarget, genai::resolver::Error> {
        Ok(ServiceTarget {
            endpoint: Endpoint::from_owned(base.clone()),
            auth: AuthData::from_single("test-key"),
            model: ModelIden::new(AdapterKind::OpenAI, t.model.model_name),
        })
    });
    Client::builder().with_service_target_resolver(resolver).build().unwrap()
}

const OK_BODY: &str = r#"{"id":"chatcmpl-1","object":"chat.completion","created":1,"model":"gpt-4o-mini",
"choices":[{"index":0,"message":{"role":"assistant","content":"Rayleigh scattering."},"finish_reason":"stop"}],
"usage":{"prompt_tokens":120000,"completion_tokens":20000,"total_tokens":140000,"prompt_tokens_details":{"cached_tokens":100000}}}"#;

#[tokio::test]
async fn exec_chat_reserves_calls_and_books_reported_usage() {
    let base = mock_server(200, OK_BODY).await;
    let client = client(base);
    let budget = AgentBudget::builder("genai-test", 1.0).build().unwrap();
    let req = ChatRequest::new(vec![ChatMessage::user("Why is the sky blue?")]);
    let opts = ChatOptions::default().with_max_tokens(100);

    let (res, outcome) = budgeted::exec_chat(&client, &budget, "gpt-4o-mini", req, Some(&opts)).await.unwrap();
    assert_eq!(res.first_text(), Some("Rayleigh scattering."));
    let expected = PriceTable::builtin()
        .cost_usage("gpt-4o-mini", &Usage::new(120_000, 20_000).with_cache_read(100_000))
        .unwrap();
    assert!((outcome.recorded_usd - expected).abs() < 1e-9, "{} vs {expected}", outcome.recorded_usd);
    assert!((budget.spent_usd() - expected).abs() < 1e-9);
    assert_eq!(budget.reserved_usd(), 0.0);
    let log = budget.drain_audit_log();
    assert_eq!(log.len(), 1);
    assert_eq!(log[0].input_tokens, 120_000);
    assert_eq!(log[0].model.as_deref(), Some("gpt-4o-mini"));
}

#[tokio::test]
async fn refused_call_is_never_sent() {
    // No server at all: if the call were sent, genai would fail to connect.
    let client = client("http://127.0.0.1:9/v1/".to_string());
    let budget = AgentBudget::builder("tiny", 0.000_01).build().unwrap();
    let req = ChatRequest::new(vec![ChatMessage::user("hello")]);
    let opts = ChatOptions::default().with_max_tokens(10_000);
    let err = budgeted::exec_chat(&client, &budget, "gpt-4o", req, Some(&opts)).await.unwrap_err();
    assert!(matches!(err, budgeted::Error::Budget(BudgetError::AgentExhausted { .. })), "{err:?}");
    assert_eq!(budget.spent_usd(), 0.0);
}

#[tokio::test]
async fn provider_error_releases_the_hold() {
    let base = mock_server(500, r#"{"error":{"message":"boom","type":"server_error"}}"#).await;
    let client = client(base);
    let budget = AgentBudget::builder("err", 1.0).build().unwrap();
    let req = ChatRequest::new(vec![ChatMessage::user("hello")]);
    let err = budgeted::exec_chat(&client, &budget, "gpt-4o-mini", req, None).await.unwrap_err();
    assert!(matches!(err, budgeted::Error::Genai(_)), "{err:?}");
    assert_eq!(budget.spent_usd(), 0.0);
    assert_eq!(budget.reserved_usd(), 0.0);
}

#[tokio::test]
async fn unknown_model_is_a_budget_error() {
    let client = client("http://127.0.0.1:9/v1/".to_string());
    let budget = AgentBudget::builder("u", 1.0).build().unwrap();
    let req = ChatRequest::new(vec![ChatMessage::user("hello")]);
    let err = budgeted::exec_chat(&client, &budget, "no-such-model", req, None).await.unwrap_err();
    assert!(matches!(err, budgeted::Error::Budget(BudgetError::UnknownModel(_))));
}
