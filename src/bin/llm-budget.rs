//! `llm-budget audit`: replay a JSON-lines log of LLM calls and show where a
//! soft limit would have let money past the cap. See `llm_budget::audit`.

use llm_budget::audit::{parse_jsonl, replay, AuditConfig};
use std::io::Read;
use std::process::ExitCode;

const USAGE: &str = "\
llm-budget audit: where would a soft limit have let money past your cap?

USAGE:
    llm-budget audit <calls.jsonl | -> --cap <USD> [OPTIONS]

One JSON object per line: a \"model\" and the provider's usage (OpenAI, Anthropic
or Gemini shape, whole response or usage block). Optional \"booked_usd\" (or
\"cost_usd\", \"cost\"): what your current tracker booked for the call.

OPTIONS:
    --cap <USD>               The hard cap to test against (required: a budget
                              without a cap is not a budget)
    --max-price-age <DAYS>    Refuse prices checked longer ago (default 90)
    --today <YYYY-MM-DD>      Judge price ages as of this date (default: today)
    --json                    Print the report as JSON
    -h, --help                Print this help

EXIT STATUS:
    0  no money past the cap and every call priced
    2  the soft limit let money past the cap, or a call had no usable price
    1  bad input or arguments
";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match run(&args) {
        Ok(code) => code,
        Err(msg) => {
            eprintln!("llm-budget: {msg}\n\n{USAGE}");
            ExitCode::from(1)
        }
    }
}

fn run(args: &[String]) -> Result<ExitCode, String> {
    if args.is_empty() || args.iter().any(|a| a == "-h" || a == "--help") {
        print!("{USAGE}");
        return Ok(ExitCode::SUCCESS);
    }
    if args[0] != "audit" {
        return Err(format!("unknown command {:?}", args[0]));
    }
    let mut path: Option<&str> = None;
    let mut cap: Option<f64> = None;
    let mut max_age: u32 = llm_budget::DEFAULT_MAX_PRICE_AGE_DAYS;
    let mut today: Option<i64> = None;
    let mut json = false;
    let mut it = args[1..].iter();
    while let Some(a) = it.next() {
        let mut value = |name: &str| it.next().map(String::as_str).ok_or(format!("{name} needs a value"));
        match a.as_str() {
            "--cap" => {
                let v = value("--cap")?;
                let x: f64 = v.trim_start_matches('$').parse().map_err(|_| format!("--cap: not a number: {v}"))?;
                if !x.is_finite() || x <= 0.0 {
                    return Err(format!("--cap must be a positive amount, got {v}"));
                }
                cap = Some(x);
            }
            "--max-price-age" => {
                let v = value("--max-price-age")?;
                max_age = v.parse().map_err(|_| format!("--max-price-age: not a whole number of days: {v}"))?;
            }
            "--today" => {
                let v = value("--today")?;
                today =
                    Some(llm_budget::safety::days_from_date(v).ok_or(format!("--today must be YYYY-MM-DD, got {v}"))?);
            }
            "--json" => json = true,
            other if other.starts_with("--") => return Err(format!("unknown option {other}")),
            other => {
                if path.replace(other).is_some() {
                    return Err("give one input file".into());
                }
            }
        }
    }
    let cap = cap.ok_or("--cap is required: a budget without a cap is not a budget")?;
    let path = path.ok_or("no input file (use - for standard input)")?;
    let text = if path == "-" {
        let mut s = String::new();
        std::io::stdin().read_to_string(&mut s).map_err(|e| format!("reading standard input: {e}"))?;
        s
    } else {
        std::fs::read_to_string(path).map_err(|e| format!("reading {path}: {e}"))?
    };
    let calls = parse_jsonl(&text).map_err(|e| format!("{path}: {e}"))?;
    let mut cfg = AuditConfig::new(cap);
    cfg.max_price_age_days = max_age;
    cfg.today = today;
    let report = replay(&calls, &cfg);
    if json {
        let out = serde_json::to_string_pretty(&report).map_err(|e| e.to_string())?;
        println!("{out}");
    } else {
        println!("llm-budget audit: {path}");
        print!("{report}");
    }
    Ok(if report.leaked() { ExitCode::from(2) } else { ExitCode::SUCCESS })
}
