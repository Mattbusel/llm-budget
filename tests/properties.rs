//! Property tests: the money invariants hold for any sequence of operations.

#![allow(clippy::unwrap_used)]
use llm_budget::budget::AtomicBudget;
use llm_budget::{AgentBudget, AuditLog, ChargeContext, FleetAllocator, ModelTier};
use std::sync::Arc;
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

    /// Whatever mix of checked charges (some refused), forced charges and
    /// reservations (settled or dropped) runs, the audit log adds up to the
    /// spend, and nothing stays reserved.
    #[test]
    fn audit_log_sums_to_spent_under_refusals(
        limit in 0.01f64..5.0,
        ops in prop::collection::vec((0u8..4, 0.0f64..2.0, 0.0f64..2.0), 0..80),
    ) {
        rt().block_on(async {
            let agent = AgentBudget::builder("p", limit).audit_log(AuditLog::Unbounded).build().unwrap();
            for (kind, a, b) in ops {
                match kind {
                    0 => {
                        let _ = agent.charge_usd(a, None, None).await; // may be refused
                    }
                    1 => {
                        agent.charge_usd_force(a, ChargeContext::new());
                    }
                    2 => {
                        if let Ok(r) = agent.reserve(a) {
                            r.settle(b);
                        }
                    }
                    _ => {
                        // reserve and drop: must leave no trace
                        let before = agent.spent_usd();
                        drop(agent.reserve(a));
                        assert_eq!(agent.spent_usd(), before);
                    }
                }
                let sum: f64 = agent.audit_log().await.iter().map(|r| r.cost_usd).sum();
                assert!((sum - agent.spent_usd()).abs() < 1e-6, "log {sum} != spent {}", agent.spent_usd());
                assert_eq!(agent.reserved_usd(), 0.0);
            }
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

/// 64 threads reserve, then settle at a random fraction of 0x to 2x their
/// estimate. Admission is atomic, so the only way past the cap is a call that
/// cost more than its reservation: the overshoot is bounded by the sum of
/// those settle deltas, and with no call over its estimate the cap holds exactly.
#[test]
fn concurrent_reservations_never_overshoot_beyond_settle_deltas() {
    for round in 0..20u64 {
        let limit = 10.0;
        let agent = Arc::new(AgentBudget::builder("hot", limit).audit_log(AuditLog::Unbounded).build().unwrap());
        let over_estimate = round % 2 == 0; // odd rounds never exceed their estimate
        let handles: Vec<_> = (0..64u64)
            .map(|t| {
                let agent = agent.clone();
                std::thread::spawn(move || {
                    // small deterministic PRNG per thread
                    let mut x = (t + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ round;
                    let mut next = move || {
                        x ^= x << 13;
                        x ^= x >> 7;
                        x ^= x << 17;
                        (x % 10_000) as f64 / 10_000.0
                    };
                    let mut overshoot_allowance = 0.0f64;
                    for _ in 0..200 {
                        let estimate = 0.001 + next() * 0.05;
                        let Ok(r) = agent.reserve(estimate) else { continue };
                        let factor = if over_estimate { next() * 2.0 } else { next() };
                        let actual = ((estimate * factor) * 1e9).round() / 1e9;
                        let held = r.estimate_usd();
                        overshoot_allowance += (actual - held).max(0.0);
                        r.settle(actual);
                    }
                    overshoot_allowance
                })
            })
            .collect();
        let allowance: f64 = handles.into_iter().map(|h| h.join().unwrap()).sum();
        let spent = agent.spent_usd();
        assert_eq!(agent.reserved_usd(), 0.0);
        assert!(spent <= limit + allowance + 1e-6, "round {round}: spent {spent} > {limit} + {allowance}");
        if !over_estimate {
            assert!(spent <= limit + 1e-9, "round {round}: spent {spent} over a hard cap of {limit}");
        }
        let log = rt().block_on(agent.audit_log());
        let sum: f64 = log.iter().map(|r| r.cost_usd).sum();
        assert!((sum - spent).abs() < 1e-6, "round {round}: log {sum} != spent {spent}");
    }
}
