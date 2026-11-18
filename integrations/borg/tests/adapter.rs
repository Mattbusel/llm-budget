//! llm-budget's borg adapter inside real borg compositions: mixed-model
//! trees, parallel admission and sub-budget exhaustion.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use borg::budget::ParallelAdmission;
use borg::{
    AgentContext, AgentLlmClient, BorgAgent, BorgError, Budget, BudgetVerdict, ChatRequest, ChatResponse,
    Composition, ExecCtx, TokenUsage,
};
use llm_budget::borg::BorgPricing;
use llm_budget::{ModelPrice, PriceTable, Usage};
use llm_budget_borg_check::{PricedBudget, PricedLlm};

/// small: $1 / $5 per million (the reference). big: 15x small on both sides.
/// mid: $3 / $10. cached: small's prices, cache reads at 10%.
fn pricing() -> Arc<BorgPricing> {
    let table = PriceTable::new()
        .with_price(ModelPrice::new("small", "test", 1.0, 5.0))
        .with_price(ModelPrice::new("big", "test", 15.0, 75.0))
        .with_price(ModelPrice::new("mid", "test", 3.0, 10.0))
        .with_price(ModelPrice::new("cached", "test", 1.0, 5.0).with_cache(Some(0.1), None));
    Arc::new(BorgPricing::with_table(Arc::new(table), "small").expect("reference is priced"))
}

/// Reference tokens of one 1k-in / 1k-out call on each model.
const SMALL: u64 = 2_000; // $0.006
const BIG: u64 = 30_000; // $0.090, 15x
const MID: u64 = 4_333; // $0.013 against $0.006, on 2,000 tokens
/// 10,000 input, 9,000 of it from cache: 1,000 + 900 tokens' worth of small.
const CACHED: u64 = 1_900;

/// Every call uses 1,000 input and 1,000 output tokens (`cached`: 10,000
/// input, 9,000 from cache, no output). A per-call delay and a yield shuffle
/// parallel branches across worker threads.
struct FakeLlm {
    calls: AtomicUsize,
}

impl FakeLlm {
    fn new() -> Arc<Self> {
        Arc::new(Self { calls: AtomicUsize::new(0) })
    }

    fn calls(&self) -> usize {
        self.calls.load(Ordering::SeqCst)
    }
}

#[async_trait]
impl AgentLlmClient for FakeLlm {
    async fn chat_completion(&self, request: ChatRequest) -> Result<ChatResponse, BorgError> {
        let n = self.calls.fetch_add(1, Ordering::SeqCst);
        tokio::time::sleep(Duration::from_millis((n % 7) as u64)).await;
        tokio::task::yield_now().await;
        Ok(match &*request.model {
            "cached" => ChatResponse::new("ok", 10_000, 0).with_cache_tokens(9_000, 0),
            _ => ChatResponse::new("ok", 1_000, 1_000),
        })
    }
}

struct Agent {
    id: &'static str,
    model: &'static str,
}

#[async_trait]
impl BorgAgent for Agent {
    fn agent_id(&self) -> &'static str {
        self.id
    }
    fn model(&self) -> &str {
        self.model
    }
    fn user_prompt(&self, _ctx: &AgentContext) -> Option<String> {
        Some("go".into())
    }
}

fn single(name: impl Into<String>, model: &'static str) -> Composition {
    Composition::single(name, Agent { id: "agent", model })
}

fn ctx(pricing: &Arc<BorgPricing>, llm: Arc<FakeLlm>, budget: Arc<dyn Budget>) -> ExecCtx {
    ExecCtx::new(
        PricedLlm::shared(pricing, llm),
        budget,
        borg::noop_emitter(),
        borg::CancellationToken::new(),
        "test",
    )
}

#[test]
fn unit_math() {
    let p = pricing();
    let u = Usage::new(1_000, 1_000);
    assert_eq!(p.units("small", &u), SMALL);
    assert_eq!(p.units("big", &u), BIG);
    assert_eq!(p.units("mid", &u), MID);
    assert_eq!(p.units("cached", &llm_budget::borg::__usage(10_000, 0, 9_000, 0)), CACHED);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_mixed_model_tree_is_charged_per_model() {
    let pricing = pricing();
    let session = PricedBudget::in_memory(&pricing, 1_000_000);
    let tree = Composition::sequential([
        single("plan", "small"),
        Composition::parallel([
            ("a", single("a", "big")),
            ("b", single("b", "small")),
            ("c", single("c", "mid")),
            ("d", single("d", "cached")),
        ]),
        single("review", "big"),
    ]);
    let llm = FakeLlm::new();
    let _out = borg::execute(tree, ctx(&pricing, llm.clone(), session.clone())).await.expect("tree runs");

    assert_eq!(llm.calls(), 6);
    assert_eq!(session.consumed(), SMALL + BIG + SMALL + MID + CACHED + BIG);
    assert_eq!(pricing.unattributed_calls(), 0, "every charge knew its model");
    let usd = 0.006 * 2.0 + 0.09 * 2.0 + 0.013 + 0.0019;
    assert!((pricing.spent_usd() - usd).abs() < 1e-9, "{}", pricing.spent_usd());
}

/// 64 parallel branches on two models, interleaved across 8 threads: no
/// call is priced on another call's model.
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn parallel_branches_keep_their_own_model() {
    let pricing = pricing();
    let session = PricedBudget::in_memory(&pricing, u64::MAX / 2);
    let branches = (0..64).map(|i| {
        let model = if i % 2 == 0 { "big" } else { "small" };
        (format!("b{i}"), single(format!("b{i}"), model))
    });
    let tree = Composition::parallel_with(branches, borg::ErrorPolicy::FailFast, 64);
    let _out = borg::execute(tree, ctx(&pricing, FakeLlm::new(), session.clone())).await.expect("runs");
    assert_eq!(session.consumed(), 32 * BIG + 32 * SMALL);
    assert_eq!(pricing.unattributed_calls(), 0);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn parallel_admission_is_in_tokens() {
    let pricing = pricing();
    // 3,500 reference tokens: borg admits parallel branches at 1,000 each.
    let session = PricedBudget::in_memory(&pricing, 3_500);
    let tree = Composition::parallel((0..5).map(|i| (format!("p{i}"), single(format!("p{i}"), "small"))));
    let llm = FakeLlm::new();
    let out = borg::execute(tree, ctx(&pricing, llm.clone(), session.clone())).await.expect("degrades");
    assert_eq!(llm.calls(), 3, "3 of 5 branches admitted");
    assert_eq!(out.verdicts.len(), 3);

    // One frontier call takes what 15 small ones would: after it, a 40k
    // session admits 8 more 1k branches, not 37.
    let session = PricedBudget::in_memory(&pricing, 40_000);
    pricing.note_call("small", Usage::new(1_000, 1_000));
    session.record_usage(&TokenUsage::new(1_000, 1_000));
    assert_eq!(session.remaining(), 38_000);
    pricing.note_call("big", Usage::new(1_000, 1_000));
    session.record_usage(&TokenUsage::new(1_000, 1_000));
    assert_eq!(session.remaining(), 8_000);
    assert_eq!(session.admit_parallel(20, 1_000), ParallelAdmission::Partial { allowed: 8 });
    assert_eq!(pricing.unattributed_calls(), 0);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_sub_budget_is_exhausted_by_price_not_token_count() {
    let pricing = pricing();
    let session = PricedBudget::in_memory(&pricing, 1_000_000);

    // A 30k pool holds exactly one big call (or fifteen small ones).
    let round = PricedBudget::sub(&pricing, session.clone(), 30_000);
    let tree = Composition::sequential([single("draft", "big"), single("revise", "small")]);
    let llm = FakeLlm::new();
    let err = borg::execute(tree, ctx(&pricing, llm.clone(), round.clone()))
        .await
        .expect_err("the pool is spent after the big call");
    match err {
        BorgError::BudgetExhausted { remaining, .. } => assert_eq!(remaining, 0),
        other => panic!("expected BudgetExhausted, got {other:?}"),
    }
    assert_eq!(llm.calls(), 1, "revise never reached the model");
    assert_eq!(round.consumed(), BIG);
    assert!(!round.can_invoke());
    assert_eq!(session.consumed(), BIG, "the session is charged the same reference tokens");
    assert!(session.can_invoke(), "the session keeps going");

    // The same size of pool runs fifteen small calls before it is spent.
    let round = PricedBudget::sub(&pricing, session.clone(), 30_000);
    let tree = Composition::sequential((0..15).map(|i| single(format!("s{i}"), "small")));
    let _out = borg::execute(tree, ctx(&pricing, FakeLlm::new(), round.clone())).await.expect("fits exactly");
    assert_eq!(round.consumed(), 30_000);
    assert!(!round.can_invoke());
    assert_eq!(session.consumed(), BIG + 30_000);
}

#[test]
fn nested_pools_and_the_parent_cap_bind() {
    let pricing = pricing();
    let session = PricedBudget::in_memory(&pricing, 100_000);
    let outer = PricedBudget::sub(&pricing, session.clone(), 40_000);
    let inner = PricedBudget::sub(&pricing, outer.clone(), 100_000);
    assert_eq!(inner.remaining(), 40_000, "capped by its parent");

    pricing.note_call("big", Usage::new(1_000, 1_000));
    assert_eq!(inner.record_usage(&TokenUsage::new(1_000, 1_000)), BudgetVerdict::Ok);
    assert_eq!((inner.consumed(), outer.consumed(), session.consumed()), (BIG, BIG, BIG));

    pricing.note_call("big", Usage::new(1_000, 1_000));
    assert_eq!(inner.record_usage(&TokenUsage::new(1_000, 1_000)), BudgetVerdict::Overdrawn);
    assert_eq!(inner.remaining(), 0);
    assert!(!outer.can_invoke());
    assert_eq!(session.remaining(), 100_000 - 2 * BIG);
    assert_eq!(pricing.unattributed_calls(), 0);
}

#[test]
fn a_charge_without_a_note_is_priced_as_the_reference_and_counted() {
    let pricing = pricing();
    let session = PricedBudget::in_memory(&pricing, 10_000);
    session.record_usage(&TokenUsage::new(1_000, 1_000));
    assert_eq!(session.consumed(), SMALL);
    assert_eq!(pricing.unattributed_calls(), 1);
}
