//! `llm-budget audit` end to end: the binary, its output and exit codes.

use std::process::Command;

fn bin() -> Command {
    Command::new(env!("CARGO_BIN_EXE_llm-budget"))
}

fn write_log(lines: &[String]) -> tempfile::NamedTempFile {
    let mut f = tempfile::NamedTempFile::new().unwrap();
    std::io::Write::write_all(&mut f, lines.join("\n").as_bytes()).unwrap();
    f
}

/// Twelve gpt-5.5 calls booked at a $1/$3 default price, the way a tracker
/// that does not know the model books them.
fn under_booked_log() -> tempfile::NamedTempFile {
    let line = r#"{"model":"gpt-5.5","usage":{"prompt_tokens":100000,"completion_tokens":5000},"booked_usd":0.115}"#;
    write_log(&vec![line.to_string(); 12])
}

#[test]
fn reports_money_past_the_cap_and_exits_2() {
    let log = under_booked_log();
    let out = bin().args(["audit", log.path().to_str().unwrap(), "--cap", "1"]).output().unwrap();
    let text = String::from_utf8(out.stdout).unwrap();
    assert_eq!(out.status.code(), Some(2), "{text}");
    assert!(text.contains("MONEY PAST THE CAP"), "{text}");
    assert!(text.contains("stopped the run at call 10"), "{text}");
    assert!(text.contains("gpt-5.5"), "{text}");
}

#[test]
fn json_report_parses() {
    let log = under_booked_log();
    let out = bin().args(["audit", log.path().to_str().unwrap(), "--cap", "1", "--json"]).output().unwrap();
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(v["calls"], 12);
    assert_eq!(v["soft"]["calls_through"], 9);
    assert!(v["soft"]["money_past_cap_usd"].as_f64().unwrap() > 4.0);
    assert!(v["hard"]["spent_usd"].as_f64().unwrap() <= 1.0);
}

#[test]
fn a_clean_log_exits_0() {
    let line = r#"{"model":"gpt-4o","usage":{"prompt_tokens":1000,"completion_tokens":100}}"#.to_string();
    let log = write_log(&[line.clone(), line]);
    let out = bin().args(["audit", log.path().to_str().unwrap(), "--cap", "$5"]).output().unwrap();
    assert_eq!(out.status.code(), Some(0), "{}", String::from_utf8_lossy(&out.stdout));
}

#[test]
fn no_cap_no_audit() {
    let log = under_booked_log();
    let out = bin().args(["audit", log.path().to_str().unwrap()]).output().unwrap();
    assert_eq!(out.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&out.stderr).contains("--cap is required"));
    let out = bin().args(["audit", log.path().to_str().unwrap(), "--cap", "0"]).output().unwrap();
    assert_eq!(out.status.code(), Some(1));
}

#[test]
fn bad_lines_are_named() {
    let log = write_log(&[
        "{\"model\":\"gpt-4o\",\"usage\":{\"prompt_tokens\":1,\"completion_tokens\":1}}".into(),
        "oops".into(),
    ]);
    let out = bin().args(["audit", log.path().to_str().unwrap(), "--cap", "1"]).output().unwrap();
    assert_eq!(out.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&out.stderr).contains("line 2"));
}

#[test]
fn help_exits_0() {
    let out = bin().arg("--help").output().unwrap();
    assert_eq!(out.status.code(), Some(0));
    assert!(String::from_utf8_lossy(&out.stdout).contains("llm-budget audit"));
}
