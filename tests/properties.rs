//! Property tests: the money invariants hold for any sequence of operations.

#![allow(clippy::unwrap_used)]
use llm_budget::budget::AtomicBudget;
use llm_budget::{AgentBudget, FleetAllocator, ModelTier};
use proptest::prelude::*;

fn rt() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_current_thread().build().unwrap()
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    /// Spent never exceeds the limit, and equals the sum of accepted charges.
    #[test]
    fn spent_is_sum_of_accepted_and_never_over_limit(
        limit in 0.01f64..100.0,
        charges in prop::collection::vec(0.0f64..10.0, 0..60),
    ) {
        let b = AtomicBudget::new(limit).unwrap();
        let mut accepted = 0.0;
        for c in charges {
            // usd amounts are kept in whole nano-dollars
            let c = (c * 1e9).round() / 1e9;
            if b.charge(c).is_ok() {
                accepted += c;
            }
            prop_assert!(b.spent_usd() <= b.limit_usd() + 1e-9);
        }
        prop_assert!((b.spent_usd() - accepted).abs() < 1e-6);
    }

    /// A refused charge leaves the counter untouched.
    #[test]
    fn refusal_changes_nothing(limit in 0.01f64..10.0, prefill in 0.0f64..1.0, extra in 0.0f64..100.0) {
        let b = AtomicBudget::new(limit).unwrap();
        let _ = b.charge(limit * prefill);
        let before = b.spent_usd();
        if b.charge(extra).is_err() {
            prop_assert_eq!(b.spent_usd(), before);
        }
    }

    /// Reallocation moves cap between agents but never creates or destroys it.
    #[test]
    fn reallocation_conserves_total(moves in prop::collection::vec((0usize..3, 0usize..3, 0.0f64..30.0), 0..30)) {
        let names = ["a", "b", "c"];
        rt().block_on(async {
            let fleet = FleetAllocator::new(90.0, ModelTier::Standard, 0.1).unwrap();
            for n in names {
                fleet.register_agent(n, 30.0).unwrap();
            }
            for (from, to, amount) in moves {
                let _ = fleet.reallocate(names[from], names[to], amount).await;
                let total: f64 = fleet.fleet_snapshot().await.iter().map(|s| s.limit_usd).sum();
                assert!((total - 90.0).abs() < 1e-6, "total cap drifted to {total}");
            }
        });
    }

    /// The receipts of all accepted charges add up to what the agent spent.
    #[test]
    fn receipts_add_up(calls in prop::collection::vec((0u32..200_000, 0u32..50_000), 0..40)) {
        rt().block_on(async {
            let agent = AgentBudget::new("p", 5.0, ModelTier::Premium, 0.2).unwrap();
            let mut sum = 0.0;
            for (i, o) in calls {
                if let Ok(r) = agent.charge_model("gpt-4o", i, o, None, None).await {
                    sum += r.cost_usd;
                }
            }
            assert!((agent.spent_usd() - sum).abs() < 1e-9);
            assert!(agent.spent_usd() <= 5.0 + 1e-9);
        });
    }

    /// Bad numbers never panic.
    #[test]
    fn odd_inputs_do_not_panic(x in prop::num::f64::ANY) {
        let _ = AtomicBudget::new(x);
        let b = AtomicBudget::new(1.0).unwrap();
        let _ = b.charge(x);
        let _ = b.shrink_limit(x);
        prop_assert!(b.spent_usd() <= b.limit_usd() + 1e-9);
    }
}
