//! `BudgetLayer`: a hard USD cap as a [`tower::Layer`] (feature `tower`).
//!
//! Wrap any tower [`Service`] that calls an LLM API (an HTTP client, a reqwest or hyper
//! service, an axum handler that proxies to a provider) and every request is priced and checked
//! against an [`AgentBudget`] **before it is sent**:
//!
//! 1. an estimator reads the request and returns the model and the expected [`Usage`]
//!    (prompt tokens plus the output you allow);
//! 2. the layer takes a [`reserve_for`](AgentBudget::reserve_for) hold for that much; if it
//!    does not fit under the cap, the call fails with [`BudgetLayerError::Refused`] and the
//!    inner service is never called;
//! 3. when the response comes back, a usage reader takes the provider's reported usage and
//!    the hold is settled at the real cost. If the reader finds no usage, the estimate is
//!    booked (the call happened). If the inner service fails, the hold is released and
//!    nothing is booked.
//!
//! For OpenAI-compatible HTTP APIs (OpenAI, OpenRouter, Groq, Together, vLLM, a gateway, ...)
//! [`BudgetLayer::openai_http`] does steps 1 and 3 for `http::Request<Bytes>` and
//! `http::Response<Bytes>`: it reads `model`, the messages and `max_tokens` /
//! `max_completion_tokens` from the request body and the `usage` block from the response.
//!
//! ```
//! use std::sync::Arc;
//! use bytes::Bytes;
//! use llm_budget::tower_layer::{BudgetLayer, BudgetLayerError};
//! use llm_budget::AgentBudget;
//! use tower::{service_fn, Layer, Service, ServiceExt};
//!
//! # #[tokio::main(flavor = "current_thread")]
//! # async fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let budget = Arc::new(AgentBudget::builder("summarizer", 0.01).build()?);
//!
//! // Stands in for your HTTP client: answers like the OpenAI API.
//! let client = service_fn(|_req: http::Request<Bytes>| async {
//!     let body = r#"{"model":"gpt-4o-mini","usage":{"prompt_tokens":1200,"completion_tokens":300}}"#;
//!     Ok::<_, std::convert::Infallible>(http::Response::new(Bytes::from(body)))
//! });
//! let mut svc = BudgetLayer::openai_http(budget.clone()).layer(client);
//!
//! let req = http::Request::post("https://api.openai.com/v1/chat/completions")
//!     .body(Bytes::from(r#"{"model":"gpt-4o-mini","max_tokens":500,
//!                          "messages":[{"role":"user","content":"Summarize this report."}]}"#))?;
//! svc.ready().await?.call(req).await?;
//! // 1,200 prompt tokens at $0.15/1M plus 300 output tokens at $0.60/1M.
//! assert!((budget.spent_usd() - 0.00036).abs() < 1e-9);
//!
//! // A request whose worst case does not fit is refused before it is sent.
//! let big = http::Request::post("https://api.openai.com/v1/chat/completions")
//!     .body(Bytes::from(r#"{"model":"gpt-4o","max_tokens":100000,"messages":[]}"#))?;
//! let err = svc.ready().await?.call(big).await.unwrap_err();
//! assert!(matches!(err, BudgetLayerError::Refused(_)));
//! # Ok(())
//! # }
//! ```

use std::fmt;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use tower::{Layer, Service};

use crate::budget::AgentBudget;
use crate::error::BudgetError;
use crate::usage::{estimate_json_tokens, Usage};

/// Reads a request: the model it calls and the usage to reserve for it.
pub type Estimator<Req> = Arc<dyn Fn(&Req) -> Result<(String, Usage), BudgetError> + Send + Sync>;

/// Reads a response: the model and usage the provider reported, if any.
pub type UsageReader<Resp> = Arc<dyn Fn(&Resp) -> Option<(String, Usage)> + Send + Sync>;

/// The error of a [`BudgetService`]: the budget refused the call, or the inner service failed.
#[derive(Debug)]
pub enum BudgetLayerError<E> {
    /// Refused before sending: the estimate did not fit under the cap, the kill switch is on,
    /// or the request could not be priced (unknown model, unreadable body).
    Refused(BudgetError),
    /// The inner service failed; the hold was released and nothing was booked.
    Inner(E),
}

impl<E: fmt::Display> fmt::Display for BudgetLayerError<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Refused(e) => write!(f, "refused by the budget: {e}"),
            Self::Inner(e) => write!(f, "{e}"),
        }
    }
}

impl<E: std::error::Error + 'static> std::error::Error for BudgetLayerError<E> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Refused(e) => Some(e),
            Self::Inner(e) => Some(e),
        }
    }
}

/// A [`Layer`] that caps the USD spend of the service it wraps; see the [module docs](self).
pub struct BudgetLayer<Req, Resp> {
    budget: Arc<AgentBudget>,
    estimate: Estimator<Req>,
    read_usage: UsageReader<Resp>,
}

impl<Req, Resp> Clone for BudgetLayer<Req, Resp> {
    fn clone(&self) -> Self {
        Self {
            budget: Arc::clone(&self.budget),
            estimate: Arc::clone(&self.estimate),
            read_usage: Arc::clone(&self.read_usage),
        }
    }
}

impl<Req, Resp> fmt::Debug for BudgetLayer<Req, Resp> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BudgetLayer").field("remaining_usd", &self.budget.remaining_usd()).finish_non_exhaustive()
    }
}

impl<Req, Resp> BudgetLayer<Req, Resp> {
    /// A layer for any request and response types: `estimate` returns the model and the usage
    /// to reserve for a request, `read_usage` the model and usage a response reports.
    pub fn new<F, G>(budget: Arc<AgentBudget>, estimate: F, read_usage: G) -> Self
    where
        F: Fn(&Req) -> Result<(String, Usage), BudgetError> + Send + Sync + 'static,
        G: Fn(&Resp) -> Option<(String, Usage)> + Send + Sync + 'static,
    {
        Self { budget, estimate: Arc::new(estimate), read_usage: Arc::new(read_usage) }
    }

    /// The budget this layer books on.
    pub fn budget(&self) -> &Arc<AgentBudget> {
        &self.budget
    }
}

/// Output tokens reserved when an OpenAI-style request sets no `max_tokens` or
/// `max_completion_tokens`.
pub const DEFAULT_MAX_OUTPUT_TOKENS: u64 = 4_096;

/// Reads an OpenAI-compatible JSON request body (Chat Completions or Responses API): the
/// `model`, an estimate of the prompt tokens (about four characters per token over every
/// string in `messages`, `input`, `instructions`, `system` and `tools`) and the output
/// allowance (`max_completion_tokens`, `max_tokens` or `max_output_tokens`, else
/// [`DEFAULT_MAX_OUTPUT_TOKENS`]).
///
/// # Errors
/// [`BudgetError::InvalidConfig`] when the body is not JSON or has no `model`.
pub fn estimate_openai_body(body: &[u8]) -> Result<(String, Usage), BudgetError> {
    let v: serde_json::Value = serde_json::from_slice(body)
        .map_err(|e| BudgetError::InvalidConfig(format!("request body is not JSON: {e}")))?;
    let model = v
        .get("model")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| BudgetError::InvalidConfig("request body has no \"model\"".into()))?
        .to_string();
    let prompt: u64 = ["messages", "input", "instructions", "system", "tools"]
        .iter()
        .filter_map(|k| v.get(*k))
        .map(estimate_json_tokens)
        .sum();
    let output = ["max_completion_tokens", "max_tokens", "max_output_tokens"]
        .iter()
        .find_map(|k| v.get(*k).and_then(serde_json::Value::as_u64))
        .unwrap_or(DEFAULT_MAX_OUTPUT_TOKENS);
    Ok((model, Usage::new(prompt, output)))
}

/// Reads the `model` and `usage` of an OpenAI-compatible JSON response body (Chat
/// Completions, Responses API, or Anthropic and Gemini shapes; see [`Usage::from_value`]).
/// `None` when the body is not JSON or has no usage block.
pub fn read_openai_body(body: &[u8]) -> Option<(String, Usage)> {
    let v: serde_json::Value = serde_json::from_slice(body).ok()?;
    let usage = Usage::from_value(&v).ok()?;
    let model = v.get("model").and_then(serde_json::Value::as_str).unwrap_or_default().to_string();
    Some((model, usage))
}

impl BudgetLayer<http::Request<bytes::Bytes>, http::Response<bytes::Bytes>> {
    /// A layer for OpenAI-compatible HTTP calls with buffered bodies: estimates with
    /// [`estimate_openai_body`] and settles with [`read_openai_body`]. For a streaming
    /// response, settle from the final usage chunk yourself with
    /// [`Reservation::settle_usage`](crate::Reservation::settle_usage) instead.
    pub fn openai_http(budget: Arc<AgentBudget>) -> Self {
        Self::new(
            budget,
            |req: &http::Request<bytes::Bytes>| estimate_openai_body(req.body()),
            |resp: &http::Response<bytes::Bytes>| read_openai_body(resp.body()),
        )
    }
}

impl<S, Req, Resp> Layer<S> for BudgetLayer<Req, Resp> {
    type Service = BudgetService<S, Req, Resp>;

    fn layer(&self, inner: S) -> Self::Service {
        BudgetService { inner, layer: self.clone() }
    }
}

/// The service a [`BudgetLayer`] wraps around `S`.
pub struct BudgetService<S, Req, Resp> {
    inner: S,
    layer: BudgetLayer<Req, Resp>,
}

impl<S: Clone, Req, Resp> Clone for BudgetService<S, Req, Resp> {
    fn clone(&self) -> Self {
        Self { inner: self.inner.clone(), layer: self.layer.clone() }
    }
}

impl<S: fmt::Debug, Req, Resp> fmt::Debug for BudgetService<S, Req, Resp> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BudgetService").field("inner", &self.inner).field("layer", &self.layer).finish()
    }
}

impl<S, Req> Service<Req> for BudgetService<S, Req, S::Response>
where
    S: Service<Req>,
    S::Future: Send + 'static,
    S::Response: Send + 'static,
    S::Error: Send + 'static,
{
    type Response = S::Response;
    type Error = BudgetLayerError<S::Error>;
    type Future = Pin<Box<dyn Future<Output = Result<S::Response, Self::Error>> + Send>>;

    fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        self.inner.poll_ready(cx).map_err(BudgetLayerError::Inner)
    }

    fn call(&mut self, req: Req) -> Self::Future {
        let hold = (self.layer.estimate)(&req)
            .and_then(|(model, estimate)| self.layer.budget.reserve_for(&model, &estimate).map(|hold| (model, hold)));
        let (model, hold) = match hold {
            Ok(h) => h,
            // Refused before the inner service sees the request.
            Err(e) => return Box::pin(std::future::ready(Err(BudgetLayerError::Refused(e)))),
        };
        let fut = self.inner.call(req);
        let read_usage = Arc::clone(&self.layer.read_usage);
        Box::pin(async move {
            match fut.await {
                Ok(resp) => {
                    match read_usage(&resp) {
                        Some((reported, usage)) => {
                            let m = if reported.is_empty() { model } else { reported };
                            let _ = hold.settle_usage(&m, &usage);
                        }
                        None => {
                            let estimate = hold.estimate_usd();
                            let _ = hold.settle(estimate);
                        }
                    }
                    Ok(resp)
                }
                // Dropping the hold releases it: a failed call books nothing.
                Err(e) => Err(BudgetLayerError::Inner(e)),
            }
        })
    }
}
