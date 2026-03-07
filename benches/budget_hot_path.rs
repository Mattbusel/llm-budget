use criterion::{black_box, criterion_group, criterion_main, Criterion};
use llm_budget::budget::AtomicBudget;
use llm_budget::fleet::FleetBudget;
use llm_budget::model::{ModelCascade, ModelTier};

fn bench_atomic_budget_charge(c: &mut Criterion) {
    let budget = AtomicBudget::new(1000.0).unwrap();
    c.bench_function("atomic_budget_charge", |b| {
        b.iter(|| {
            black_box(budget.charge(0.000001).unwrap())
        })
    });
}

fn bench_fleet_budget_allocate(c: &mut Criterion) {
    c.bench_function("fleet_budget_allocate", |b| {
        b.iter(|| {
            let fleet = FleetBudget::new(1_000_000.0).unwrap();
            black_box(fleet.allocate("agent", 1.0).unwrap())
        })
    });
}

fn bench_fleet_budget_available(c: &mut Criterion) {
    let fleet = FleetBudget::new(100.0).unwrap();
    fleet.allocate("a1", 10.0).unwrap();
    c.bench_function("fleet_budget_available_usd", |b| {
        b.iter(|| {
            black_box(fleet.available_usd())
        })
    });
}

fn bench_model_cascade_evaluate(c: &mut Criterion) {
    let mut cascade = ModelCascade::new(ModelTier::Premium, 0.2, 100.0).unwrap();
    c.bench_function("model_cascade_evaluate", |b| {
        b.iter(|| {
            // evaluate with plenty of budget remaining — no-op fast path
            black_box(cascade.evaluate(80.0))
        })
    });
}

criterion_group!(
    benches,
    bench_atomic_budget_charge,
    bench_fleet_budget_allocate,
    bench_fleet_budget_available,
    bench_model_cascade_evaluate,
);
criterion_main!(benches);
