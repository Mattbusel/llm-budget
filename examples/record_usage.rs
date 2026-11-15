//! The core pattern with no framework and no async runtime: reserve before a
//! call, settle with the provider's reported usage after it, and record calls
//! that were made without a reservation.
//!
//! `fake_call` stands in for any client (reqwest, an SDK, a subprocess); all
//! the budget needs from it is the token counts it reports.
//!
//! Run with `cargo run --example record_usage`.

use llm_budget::{AgentBudget, BudgetError, ChargeContext, ModelPrice, PriceTable, Usage};

/// Pretend provider call: returns (uncached input, cache read, cache write, output).
fn fake_call(turn: u64) -> (u64, u64, u64, u64) {
    (400 + 50 * turn, 12_000, if turn == 0 { 12_000 } else { 0 }, 300)
}

fn main() -> Result<(), BudgetError> {
    // Your own price for a self-hosted model, on top of the built-in table.
    let prices = PriceTable::builtin()
        .clone()
        .with_price(ModelPrice::new("in-house-70b", "self-hosted", 0.20, 0.60));
    let agent = AgentBudget::builder("researcher", 0.25).price_table(prices).build()?;

    let model = "claude-sonnet-4-6";
    for turn in 0.. {
        // 1. Before the call: hold the worst case (prompt so far plus the
        //    output limit). Refused means the call is never made.
        let estimate = Usage::new(25_000, 1_000);
        let hold = match agent.reserve_for(model, &estimate) {
            Ok(hold) => hold,
            Err(e) => {
                println!("turn {turn}: not sent ({e})");
                break;
            }
        };

        // 2. The call itself, with any client.
        let (fresh, read, write, out) = fake_call(turn);

        // 3. After the call: book exactly what was reported.
        let usage = Usage::from_parts(fresh, read, write, out);
        let outcome = hold.settle_usage_with(model, &usage, ChargeContext::new().task(format!("turn-{turn}")));
        println!(
            "turn {turn}: ${:.5} (total ${:.5}, ${:.5} left)",
            outcome.recorded_usd,
            outcome.total_spent_usd,
            agent.remaining_usd()
        );
        if outcome.over_cap {
            break;
        }
    }

    // A call made elsewhere (a background job, another library) that you only
    // hear about afterwards: record it. It is booked even past the cap.
    let late = agent.record("in-house-70b", &Usage::new(50_000, 2_000))?;
    println!("late call on in-house-70b: ${:.5}, over cap: {}", late.recorded_usd, late.over_cap);

    let total: f64 = agent.drain_audit_log().iter().map(|r| r.cost_usd).sum();
    println!("audit log total ${total:.5} = spent ${:.5}", agent.spent_usd());
    Ok(())
}
