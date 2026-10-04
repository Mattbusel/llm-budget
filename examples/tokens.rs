//! Count tokens from text. Run with `cargo run --example tokens --features tokens`.

use llm_budget::tokens::{cost_of_text, count_tokens};

fn main() {
    let n = count_tokens("gpt-4o", "Hello, world!");
    assert_eq!(n.tokens, 4); // exact: gpt-4o's own tokenizer
    let c = count_tokens("claude-sonnet-4-5", "Hello, world!");
    assert!(!c.exact); // Anthropic's tokenizer is not public, so this is an estimate
    println!("gpt-4o: {} tokens (exact: {})", n.tokens, n.exact);
    println!("claude-sonnet-4-5: {} tokens (exact: {})", c.tokens, c.exact);

    let prompt = "Summarise the plot of Hamlet in one sentence.";
    let answer = "A prince avenges his father's murder and almost everyone dies.";
    match cost_of_text("gpt-4o-mini", prompt, answer) {
        Ok(usd) => println!("gpt-4o-mini would charge ${usd:.8} for that exchange"),
        Err(e) => println!("error: {e}"),
    }
}
