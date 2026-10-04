//! The charge hot path against the alternatives someone would otherwise use:
//!
//! - `llm-budget 0.2` `AtomicBudget::charge` (compare-and-swap loop)
//! - `llm-budget 0.1.1` `AtomicBudget::charge` (add, then roll back if over)
//! - a `std::sync::Mutex<u64>` check-and-add (the obvious hand-written version)
//! - `governor 0.10` `check_n` on a direct limiter (lock-free admission control
//!   from the most used Rust rate limiter; a different question, same shape)
//!
//! Each case admits charges of 1 unit against a limit large enough never to
//! refuse, so the numbers measure bookkeeping cost, not refusals. The
//! contended case runs 8 threads doing 10,000 charges each.
//!
//! Run: `cargo bench --bench vs_alternatives`
#![allow(clippy::unwrap_used, clippy::expect_used)]

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use governor::{Quota, RateLimiter};
use std::num::NonZeroU32;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

const THREADS: usize = 8;
const PER_THREAD: usize = 10_000;

trait Charger: Send + Sync + 'static {
    fn charge(&self) -> bool;
}

impl Charger for llm_budget::budget::AtomicBudget {
    fn charge(&self) -> bool {
        llm_budget::budget::AtomicBudget::charge(self, 0.000_001).is_ok()
    }
}

struct Old(llm_budget_011::budget::AtomicBudget);
impl Charger for Old {
    fn charge(&self) -> bool {
        self.0.charge(0.000_001).is_ok()
    }
}

struct Locked {
    spent: Mutex<u64>,
    limit: u64,
}
impl Charger for Locked {
    fn charge(&self) -> bool {
        let mut s = self.spent.lock().unwrap();
        if *s + 1_000 > self.limit {
            return false;
        }
        *s += 1_000;
        true
    }
}

struct Gov(governor::DefaultDirectRateLimiter);
impl Charger for Gov {
    fn charge(&self) -> bool {
        matches!(self.0.check_n(NonZeroU32::new(1).unwrap()), Ok(Ok(())))
    }
}

fn contenders() -> Vec<(&'static str, Arc<dyn Charger>)> {
    let big = 1.0e9;
    vec![
        ("llm-budget 0.2 (CAS)", Arc::new(llm_budget::budget::AtomicBudget::new(big).unwrap())),
        ("llm-budget 0.1.1 (add+rollback)", Arc::new(Old(llm_budget_011::budget::AtomicBudget::new(big).unwrap()))),
        ("Mutex<u64>", Arc::new(Locked { spent: Mutex::new(0), limit: u64::MAX / 2 })),
        (
            "governor check_n",
            Arc::new(Gov(RateLimiter::direct(Quota::per_second(NonZeroU32::new(u32::MAX).unwrap())))),
        ),
    ]
}

fn single_thread(c: &mut Criterion) {
    let mut g = c.benchmark_group("charge/1 thread");
    for (name, ch) in contenders() {
        g.bench_function(name, |b| b.iter(|| black_box(ch.charge())));
    }
    g.finish();
}

fn contended(c: &mut Criterion) {
    let mut g = c.benchmark_group("charge/8 threads x 10k");
    g.sample_size(20);
    for (name, ch) in contenders() {
        g.bench_function(name, |b| {
            b.iter_custom(|iters| {
                let mut total = Duration::ZERO;
                for _ in 0..iters {
                    let start = Instant::now();
                    let handles: Vec<_> = (0..THREADS)
                        .map(|_| {
                            let ch = Arc::clone(&ch);
                            std::thread::spawn(move || {
                                for _ in 0..PER_THREAD {
                                    black_box(ch.charge());
                                }
                            })
                        })
                        .collect();
                    for h in handles {
                        h.join().unwrap();
                    }
                    total += start.elapsed();
                }
                total
            })
        });
    }
    g.finish();
}

criterion_group!(benches, single_thread, contended);
criterion_main!(benches);
