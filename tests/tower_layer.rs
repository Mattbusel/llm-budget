//! BudgetLayer: refuses before send, settles the reported usage, releases on failure.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use bytes::Bytes;
use llm_budget::tower_layer::{estimate_openai_body, read_openai_body, BudgetLayer, BudgetLayerError};
use llm_budget::{AgentBudget, BudgetError, Usage};
use tower::{service_fn, Layer, Service, ServiceExt};

const OK_BODY: &str = r#"{"id":"x","model":"gpt-4o-mini-2024-07-18","usage":{"prompt_tokens":1000,"completion_tokens":200,"total_tokens":1200}}"#;

fn chat(model: &str, max_tokens: u64, content: &str) -> http::Request<Bytes> {
    let body = serde_json::json!({
        "model": model,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": content}],
    });
    http::Request::post("https://example.test/v1/chat/completions").body(Bytes::from(body.to_string())).unwrap()
}

#[tokio::test]
async fn request_over_the_cap_is_refused_before_send() {
    let sent = Arc::new(AtomicUsize::new(0));
    let counter = Arc::clone(&sent);
    let client = service_fn(move |_req: http::Request<Bytes>| {
        counter.fetch_add(1, Ordering::SeqCst);
        async { Ok::<_, std::convert::Infallible>(http::Response::new(Bytes::from(OK_BODY))) }
    });
    let budget = Arc::new(AgentBudget::builder("t", 0.05).build().unwrap());
    let mut svc = BudgetLayer::openai_http(Arc::clone(&budget)).layer(client);

    // gpt-4o output is $10 per 1M: 10,000 output tokens is $0.10, over the $0.05 cap.
    let err = svc.ready().await.unwrap().call(chat("gpt-4o", 10_000, "hi")).await.unwrap_err();
    match err {
        BudgetLayerError::Refused(BudgetError::AgentExhausted { .. }) => {}
        other => panic!("expected a refusal, got {other:?}"),
    }
    assert_eq!(sent.load(Ordering::SeqCst), 0, "the inner service must not be called");
    assert_eq!(budget.spent_usd(), 0.0);
    assert_eq!(budget.reserved_usd(), 0.0);

    // A call that fits goes through and books the reported usage, priced on the dated model
    // id the response names (gpt-4o-mini: $0.15 in, $0.60 out per 1M).
    svc.ready().await.unwrap().call(chat("gpt-4o-mini", 500, "hi")).await.unwrap();
    assert_eq!(sent.load(Ordering::SeqCst), 1);
    assert!((budget.spent_usd() - (1000.0 * 0.15 + 200.0 * 0.60) / 1e6).abs() < 1e-12);
    assert_eq!(budget.reserved_usd(), 0.0);
}

#[tokio::test]
async fn a_failed_call_books_nothing() {
    let client = service_fn(|_req: http::Request<Bytes>| async { Err::<http::Response<Bytes>, _>("upstream down") });
    let budget = Arc::new(AgentBudget::builder("t", 1.0).build().unwrap());
    let mut svc = BudgetLayer::openai_http(Arc::clone(&budget)).layer(client);
    let err = svc.ready().await.unwrap().call(chat("gpt-4o-mini", 100, "hi")).await.unwrap_err();
    assert!(matches!(err, BudgetLayerError::Inner("upstream down")));
    assert_eq!(budget.spent_usd(), 0.0);
    assert_eq!(budget.reserved_usd(), 0.0);
}

#[tokio::test]
async fn no_usage_in_the_response_books_the_estimate() {
    let client = service_fn(|_req: http::Request<Bytes>| async {
        Ok::<_, std::convert::Infallible>(http::Response::new(Bytes::from_static(b"not json")))
    });
    let budget = Arc::new(AgentBudget::builder("t", 1.0).build().unwrap());
    let mut svc = BudgetLayer::openai_http(Arc::clone(&budget)).layer(client);
    let req = chat("gpt-4o-mini", 1_000, "abcdefgh");
    let (model, est) = estimate_openai_body(req.body()).unwrap();
    assert_eq!(model, "gpt-4o-mini");
    // "user" is 1 token, "abcdefgh" 2, plus the 1,000 output tokens allowed.
    assert_eq!(est, Usage::new(3, 1_000));
    svc.ready().await.unwrap().call(req).await.unwrap();
    let want = (3.0 * 0.15 + 1_000.0 * 0.60) / 1e6;
    assert!((budget.spent_usd() - want).abs() < 1e-12, "{}", budget.spent_usd());
}

#[tokio::test]
async fn unreadable_or_unknown_requests_are_refused() {
    let client = service_fn(|_req: http::Request<Bytes>| async {
        Ok::<_, std::convert::Infallible>(http::Response::new(Bytes::from(OK_BODY)))
    });
    let budget = Arc::new(AgentBudget::builder("t", 1.0).build().unwrap());
    let mut svc = BudgetLayer::openai_http(budget).layer(client);
    let no_model = http::Request::post("/").body(Bytes::from_static(b"{\"messages\":[]}")).unwrap();
    assert!(matches!(svc.ready().await.unwrap().call(no_model).await, Err(BudgetLayerError::Refused(_))));
    let unknown = chat("no-such-model-xyz", 10, "hi");
    assert!(matches!(
        svc.ready().await.unwrap().call(unknown).await,
        Err(BudgetLayerError::Refused(BudgetError::UnknownModel(_)))
    ));
}

#[tokio::test]
async fn any_request_type_with_a_custom_estimator() {
    #[derive(Debug)]
    struct Ask {
        model: &'static str,
        prompt_tokens: u64,
    }
    struct Answer {
        usage: Usage,
    }
    let client = service_fn(|req: Ask| async move {
        Ok::<_, std::convert::Infallible>(Answer { usage: Usage::new(req.prompt_tokens, 50) })
    });
    let budget = Arc::new(AgentBudget::builder("t", 1.0).build().unwrap());
    let layer = BudgetLayer::new(
        Arc::clone(&budget),
        |r: &Ask| Ok((r.model.to_string(), Usage::new(r.prompt_tokens, 100))),
        |a: &Answer| Some((String::new(), a.usage)),
    );
    let mut svc = layer.layer(client);
    svc.ready().await.unwrap().call(Ask { model: "claude-sonnet-4-6", prompt_tokens: 2_000 }).await.unwrap();
    // Claude Sonnet 4.6: $3 in, $15 out per 1M; the empty reported model falls back to the request's.
    assert!((budget.spent_usd() - (2_000.0 * 3.0 + 50.0 * 15.0) / 1e6).abs() < 1e-12);
}

#[test]
fn reads_responses_api_and_anthropic_usage() {
    let r = read_openai_body(br#"{"model":"gpt-5","usage":{"input_tokens":10,"input_tokens_details":{"cached_tokens":4},"output_tokens":3}}"#).unwrap();
    assert_eq!(r.0, "gpt-5");
    assert_eq!((r.1.input_tokens, r.1.cached_input_tokens, r.1.output_tokens), (10, 4, 3));
    let a = read_openai_body(br#"{"model":"claude-haiku-4-5","usage":{"input_tokens":5,"output_tokens":7}}"#).unwrap();
    assert_eq!((a.1.input_tokens, a.1.output_tokens), (5, 7));
    assert!(read_openai_body(b"{}").is_none());
}
