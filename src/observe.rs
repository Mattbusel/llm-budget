//! Optional observability hooks. Each function is a no-op unless the
//! `tracing` and/or `metrics` feature is on.
//!
//! Metrics (feature `metrics`, through the [metrics](https://crates.io/crates/metrics)
//! facade, so any exporter works: Prometheus, StatsD, OpenTelemetry, ...):
//!
//! | Name | Type | Labels | Meaning |
//! |---|---|---|---|
//! | `llm_budget_charges_total` | counter | `agent`, `outcome` (`accepted`, `refused`, `killed`) | charge attempts |
//! | `llm_budget_spent_usd` | gauge | `agent` | total spent by the agent |
//! | `llm_budget_remaining_usd` | gauge | `agent` | budget left |
//! | `llm_budget_downgrades_total` | counter | `agent`, `tier` | tier downgrades |
//!
//! Tracing events (feature `tracing`, target `llm_budget`): `debug` for each
//! accepted charge, `warn` for refusals and kills, `info` for downgrades.

#![allow(unused_variables)]

use crate::model::ModelTier;

pub(crate) fn charged(agent: &str, model: Option<&str>, cost_usd: f64, total_usd: f64, remaining_usd: f64) {
    #[cfg(feature = "tracing")]
    tracing::debug!(target: "llm_budget", agent, model, cost_usd, total_usd, remaining_usd, "charge accepted");
    #[cfg(feature = "metrics")]
    {
        let agent = agent.to_string();
        metrics::counter!("llm_budget_charges_total", "agent" => agent.clone(), "outcome" => "accepted").increment(1);
        metrics::gauge!("llm_budget_spent_usd", "agent" => agent.clone()).set(total_usd);
        metrics::gauge!("llm_budget_remaining_usd", "agent" => agent).set(remaining_usd);
    }
}

pub(crate) fn refused(agent: &str, killed: bool, reason: &str) {
    #[cfg(feature = "tracing")]
    tracing::warn!(target: "llm_budget", agent, killed, reason, "charge refused");
    #[cfg(feature = "metrics")]
    {
        let outcome = if killed { "killed" } else { "refused" };
        metrics::counter!("llm_budget_charges_total", "agent" => agent.to_string(), "outcome" => outcome).increment(1);
    }
}

pub(crate) fn downgraded(agent: &str, tier: ModelTier) {
    #[cfg(feature = "tracing")]
    tracing::info!(target: "llm_budget", agent, tier = ?tier, "model tier downgraded");
    #[cfg(feature = "metrics")]
    metrics::counter!("llm_budget_downgrades_total", "agent" => agent.to_string(), "tier" => format!("{tier:?}")).increment(1);
}

pub(crate) fn killed(agent: &str, reason: &str) {
    #[cfg(feature = "tracing")]
    tracing::warn!(target: "llm_budget", agent, reason, "kill switch activated");
}

#[cfg(all(test, feature = "metrics"))]
mod tests {
    use crate::{AgentBudget, BudgetError, ModelTier};
    use metrics_util::debugging::{DebugValue, DebuggingRecorder};

    #[test]
    fn metrics_are_recorded() {
        let recorder = DebuggingRecorder::new();
        let snapshotter = recorder.snapshotter();
        let rt = tokio::runtime::Builder::new_current_thread().build().unwrap();
        metrics::with_local_recorder(&recorder, || {
            rt.block_on(async {
                let agent = AgentBudget::new("m1", 1.0, ModelTier::Premium, 0.5).unwrap();
                agent.charge_usd(0.6, None, None).await.unwrap(); // downgrade at 40% left
                assert!(matches!(agent.charge_usd(0.6, None, None).await, Err(BudgetError::AgentExhausted { .. })));
                agent.kill("stop");
                assert!(agent.charge_usd(0.1, None, None).await.is_err());
            })
        });
        let snap = snapshotter.snapshot().into_vec();
        let find = |name: &str, label: Option<(&str, &str)>| {
            snap.iter()
                .find(|(k, _, _, _)| {
                    k.key().name() == name
                        && label.is_none_or(|(lk, lv)| k.key().labels().any(|l| l.key() == lk && l.value() == lv))
                })
                .map(|(_, _, _, v)| v)
        };
        assert_eq!(find("llm_budget_charges_total", Some(("outcome", "accepted"))), Some(&DebugValue::Counter(1)));
        assert_eq!(find("llm_budget_charges_total", Some(("outcome", "refused"))), Some(&DebugValue::Counter(1)));
        assert_eq!(find("llm_budget_charges_total", Some(("outcome", "killed"))), Some(&DebugValue::Counter(1)));
        assert_eq!(find("llm_budget_downgrades_total", Some(("tier", "Standard"))), Some(&DebugValue::Counter(1)));
        match find("llm_budget_spent_usd", None) {
            Some(DebugValue::Gauge(g)) => assert!((g.into_inner() - 0.6).abs() < 1e-9),
            other => panic!("spent gauge missing: {other:?}"),
        }
    }
}
