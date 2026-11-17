//! Property test: random sequences of child scopes, reservations,
//! settlements, forced charges and closes keep every scope's books exact.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::sync::Arc;

use llm_budget::{AgentBudget, ChargeContext, Limit, Reservation, ScopeGuard};
use proptest::prelude::*;

#[derive(Debug, Clone)]
enum Op {
    /// Child of scope `parent % n` with this many cents, or a fraction.
    Child { parent: usize, cents: u32, fraction: bool },
    /// Reserve this many cents on scope `on % n`.
    Reserve { on: usize, cents: u32 },
    /// Settle hold `hold % h` at `pct`% of its estimate (above 100 = overrun).
    Settle { hold: usize, pct: u32 },
    /// Drop hold `hold % h` unsettled.
    Release { hold: usize },
    /// Book this many cents on scope `on % n` whatever the cap.
    Force { on: usize, cents: u32 },
    /// Close child scope `scope % (n - 1)`.
    Close { scope: usize },
}

fn op() -> impl Strategy<Value = Op> {
    prop_oneof![
        (any::<usize>(), 1u32..60, any::<bool>()).prop_map(|(parent, cents, fraction)| Op::Child { parent, cents, fraction }),
        (any::<usize>(), 0u32..30).prop_map(|(on, cents)| Op::Reserve { on, cents }),
        (any::<usize>(), 0u32..130).prop_map(|(hold, pct)| Op::Settle { hold, pct }),
        any::<usize>().prop_map(|hold| Op::Release { hold }),
        (any::<usize>(), 0u32..10).prop_map(|(on, cents)| Op::Force { on, cents }),
        any::<usize>().prop_map(|scope| Op::Close { scope }),
    ]
}

fn nanos(usd: f64) -> u64 {
    (usd * 1e9).round() as u64
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(512))]
    #[test]
    fn books_stay_exact(ops in prop::collection::vec(op(), 1..80)) {
        let root = Arc::new(AgentBudget::builder("root", 1.00).max_depth(4).build().unwrap());
        // scopes[0] is the root; guards[i] closes scopes[i + 1].
        let mut scopes: Vec<Arc<AgentBudget>> = vec![root.clone()];
        let mut guards: Vec<Option<ScopeGuard>> = Vec::new();
        let mut holds: Vec<Option<Reservation>> = Vec::new();
        let mut booked: u64 = 0; // everything booked anywhere, in nanos
        let mut overrun = false;

        for op in ops {
            let n = scopes.len();
            match op {
                Op::Child { parent, cents, fraction } => {
                    let limit = if fraction { Limit::Fraction(f64::from(cents) / 100.0) } else { Limit::Usd(f64::from(cents) / 100.0) };
                    if let Ok(g) = scopes[parent % n].child(format!("s{n}"), limit) {
                        scopes.push(g.budget().clone());
                        guards.push(Some(g));
                    }
                }
                Op::Reserve { on, cents } => {
                    if let Ok(h) = scopes[on % n].reserve(f64::from(cents) / 100.0) {
                        holds.push(Some(h));
                    }
                }
                Op::Settle { hold, pct } => {
                    if holds.is_empty() { continue; }
                    let i = hold % holds.len();
                    if let Some(h) = holds[i].take() {
                        let actual = h.estimate_usd() * f64::from(pct) / 100.0;
                        if pct > 100 { overrun = true; }
                        let out = h.settle(actual);
                        booked += nanos(out.recorded_usd);
                    }
                }
                Op::Release { hold } => {
                    if holds.is_empty() { continue; }
                    let i = hold % holds.len();
                    drop(holds[i].take());
                }
                Op::Force { on, cents } => {
                    if cents > 0 { overrun = true; }
                    let out = scopes[on % n].charge_usd_force(f64::from(cents) / 100.0, ChargeContext::new());
                    booked += nanos(out.recorded_usd);
                }
                Op::Close { scope } => {
                    if guards.is_empty() { continue; }
                    let i = scope % guards.len();
                    if let Some(g) = guards[i].take() {
                        let r = g.settle();
                        prop_assert!((r.own_cost_usd + r.child_cost_usd - r.spent_usd).abs() < 1e-9);
                    }
                }
            }
            // Root spend is everything booked anywhere in the tree.
            prop_assert!((root.spent_usd() - booked as f64 / 1e9).abs() < 1e-9,
                "root {} vs booked {}", root.spent_usd(), booked as f64 / 1e9);
            for s in &scopes {
                prop_assert!((s.own_cost_usd() + s.child_cost_usd() - s.spent_usd()).abs() < 1e-9);
                if !overrun {
                    prop_assert!(s.utilization() <= 1.0 + 1e-12, "{} at {}", s.agent_id, s.utilization());
                }
            }
        }

        // Settle nothing more: drop every hold and close every scope.
        holds.clear();
        guards.clear();
        prop_assert!(root.reserved_usd().abs() < 1e-12, "left reserved: {}", root.reserved_usd());
        prop_assert!((root.spent_usd() - booked as f64 / 1e9).abs() < 1e-9);
        prop_assert!((root.remaining_usd() - (1.0 - root.spent_usd()).max(0.0)).abs() < 1e-9);
    }
}
