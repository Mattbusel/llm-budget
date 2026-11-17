//! Child budget scopes under parallel fan-out: the overshoot of a shared
//! "remaining" figure, and the reserve/settle invariant under concurrency.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Barrier};
use std::thread;

use llm_budget::{AgentBudget, BudgetError, ChargeContext, Limit, ScopeGuard};

fn near(a: f64, b: f64) -> bool {
    (a - b).abs() < 1e-9
}

/// The pattern the A2A budget draft starts from: the parent tells each
/// delegate how much is left, and each delegate enforces that figure on its
/// own. Two delegates started in parallel both see the same $1.00.
#[test]
fn propagated_remainder_overshoots_with_two_parallel_delegates() {
    let parent = Arc::new(AgentBudget::builder("parent", 1.00).build().unwrap());
    let start = Arc::new(Barrier::new(2));
    let handles: Vec<_> = (0..2)
        .map(|i| {
            let parent = parent.clone();
            let start = start.clone();
            thread::spawn(move || {
                // Each delegate reads "remaining" before either has spent.
                let remaining = parent.remaining_usd();
                start.wait();
                let own = AgentBudget::builder(format!("delegate-{i}"), remaining).build().unwrap();
                // Four calls of $0.20, each checked against its own copy of the remainder.
                for _ in 0..4 {
                    let hold = own.reserve(0.20).unwrap();
                    hold.settle(0.20);
                    parent.charge_usd_force(0.20, ChargeContext::new());
                }
            })
        })
        .collect();
    for h in handles {
        h.join().unwrap();
    }
    // $1.60 booked against a $1.00 cap: 60% over.
    assert!(near(parent.spent_usd(), 1.60), "{}", parent.spent_usd());
}

/// The same two delegates with child scopes: each reserves its slice from
/// the parent first, so the second can only get what is left.
#[test]
fn child_scopes_hold_the_cap_with_two_parallel_delegates() {
    let parent = Arc::new(AgentBudget::builder("parent", 1.00).build().unwrap());
    let start = Arc::new(Barrier::new(2));
    let handles: Vec<_> = (0..2)
        .map(|i| {
            let parent = parent.clone();
            let start = start.clone();
            thread::spawn(move || {
                start.wait();
                // Both ask for $0.80, the most either would spend.
                let Ok(scope) = parent.child(format!("delegate-{i}"), Limit::Usd(0.80)) else {
                    return 0;
                };
                let mut calls = 0;
                while let Ok(hold) = scope.reserve(0.20) {
                    hold.settle(0.20);
                    calls += 1;
                }
                calls
            })
        })
        .collect();
    let calls: Vec<i32> = handles.into_iter().map(|h| h.join().unwrap()).collect();
    assert!(parent.spent_usd() <= 1.00 + 1e-9, "{}", parent.spent_usd());
    // One delegate got its $0.80, the other was refused up front.
    let mut sorted = calls.clone();
    sorted.sort();
    assert_eq!(sorted, vec![0, 4]);
    assert!(near(parent.reserved_usd(), 0.0));
}

/// Fractions instead of fixed amounts: each delegate takes half of what is
/// left when it starts. Whatever the interleaving, the total stays capped.
#[test]
fn fraction_slices_in_parallel_stay_under_the_cap() {
    for _ in 0..50 {
        let parent = Arc::new(AgentBudget::builder("parent", 1.00).build().unwrap());
        let start = Arc::new(Barrier::new(8));
        let handles: Vec<_> = (0..8)
            .map(|i| {
                let parent = parent.clone();
                let start = start.clone();
                thread::spawn(move || {
                    start.wait();
                    if let Ok(scope) = parent.child(format!("d{i}"), Limit::Fraction(0.5)) {
                        let all = scope.limit_usd();
                        scope.charge_usd_force(all, ChargeContext::new());
                    }
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
        assert!(parent.spent_usd() <= 1.00 + 1e-9);
        assert!(near(parent.reserved_usd(), 0.0));
    }
}

/// 64 children, each running its own calls and a grandchild, all at once,
/// while a monitor thread checks the parent never commits past its cap.
#[test]
fn sixty_four_parallel_children_never_overcommit_the_parent() {
    let parent = Arc::new(AgentBudget::builder("parent", 10.00).build().unwrap());
    let stop = Arc::new(AtomicBool::new(false));
    let monitor = {
        let parent = parent.clone();
        let stop = stop.clone();
        thread::spawn(move || {
            let mut samples = 0u64;
            while !stop.load(Ordering::SeqCst) {
                // spend plus open holds, read in one load
                let u = parent.utilization();
                assert!(u <= 1.0 + 1e-12, "parent overcommitted: {u}");
                samples += 1;
            }
            samples
        })
    };
    let handles: Vec<_> = (0..64)
        .map(|i| {
            let parent = parent.clone();
            thread::spawn(move || -> f64 {
                // Ask for $0.25 each: 64 x 0.25 = $16 wanted from a $10 cap.
                let Ok(scope) = parent.child(format!("c{i}"), Limit::Usd(0.25)) else {
                    return 0.0;
                };
                let mut own = 0.0;
                for k in 0..20 {
                    let Ok(hold) = scope.reserve(0.01) else { break };
                    // actual cost below the estimate, varying per call
                    let actual = 0.005 + f64::from((i + k) % 5) * 0.001;
                    hold.settle(actual);
                    own += actual;
                }
                if let Ok(g) = scope.child(format!("g{i}"), Limit::Fraction(0.5)) {
                    while let Ok(hold) = g.reserve(0.02) {
                        hold.settle(0.015);
                    }
                }
                let receipt = scope.settle();
                assert!(!receipt.over_hold);
                assert!((receipt.own_cost_usd - own).abs() < 1e-6);
                receipt.spent_usd
            })
        })
        .collect();
    let spent: f64 = handles.into_iter().map(|h| h.join().unwrap()).sum();
    stop.store(true, Ordering::SeqCst);
    let samples = monitor.join().unwrap();
    assert!(samples > 0);
    assert!((parent.spent_usd() - spent).abs() < 1e-6, "{} vs {spent}", parent.spent_usd());
    assert!(parent.spent_usd() <= 10.00 + 1e-9);
    assert!(near(parent.reserved_usd(), 0.0));
    assert!((parent.child_cost_usd() - parent.spent_usd()).abs() < 1e-9);
}

/// A sub-agent that panics drops its guard while unwinding: its hold goes
/// back to the parent and what it already spent stays booked.
#[test]
fn a_panicking_sub_agent_refunds_its_hold() {
    let parent = Arc::new(AgentBudget::builder("parent", 1.00).build().unwrap());
    let scope = parent.child("flaky", Limit::Usd(0.50)).unwrap();
    let h = thread::spawn(move || {
        scope.reserve(0.10).unwrap().settle(0.10);
        let _in_flight = scope.reserve(0.20).unwrap();
        panic!("sub-agent crashed");
    });
    assert!(h.join().is_err());
    assert!(near(parent.spent_usd(), 0.10));
    assert!(near(parent.reserved_usd(), 0.0));
    assert!(near(parent.remaining_usd(), 0.90));
}

/// Four sub-agents on tokio, one parent cap; one is cancelled mid-call.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn tokio_fan_out_with_a_cancelled_task() {
    let parent = Arc::new(AgentBudget::builder("orchestrator", 1.00).build().unwrap());
    let mut tasks = Vec::new();
    for i in 0..4 {
        let scope: ScopeGuard = parent.child(format!("sub-{i}"), Limit::Usd(0.25)).unwrap();
        tasks.push(tokio::spawn(async move {
            loop {
                let hold = match scope.reserve(0.04) {
                    Ok(h) => h,
                    Err(_) => break,
                };
                tokio::task::yield_now().await;
                if i == 3 {
                    // never finishes: aborted from outside while holding $0.04
                    std::future::pending::<()>().await;
                }
                hold.settle(0.03);
            }
            scope.settle()
        }));
    }
    let cancelled = tasks.pop().unwrap();
    tokio::task::yield_now().await;
    cancelled.abort();
    assert!(cancelled.await.unwrap_err().is_cancelled());
    for t in tasks {
        let r = t.await.unwrap();
        // each $0.04 hold settles at $0.03: eight calls fit in $0.25
        assert!(near(r.spent_usd, 0.24), "{r:?}");
        assert!(near(r.released_usd, 0.01), "{r:?}");
    }
    assert!(near(parent.spent_usd(), 0.72));
    assert!(near(parent.reserved_usd(), 0.0));
}

#[test]
fn depth_exceeded_is_typed() {
    let root = AgentBudget::builder("root", 1.0).max_depth(1).build().unwrap();
    let a = root.child("a", Limit::Usd(0.5)).unwrap();
    match a.child("b", Limit::Usd(0.1)) {
        Err(BudgetError::DepthExceeded { depth: 2, max_depth: 1, .. }) => {}
        other => panic!("{other:?}"),
    }
}
