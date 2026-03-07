//! Model tier definitions and automatic downgrade cascade.
//!
//! When budget pressure is detected, the cascade selects the next cheaper tier
//! automatically. The cascade is: Premium → Standard → Economy → Local.

/// Cost tier of an LLM model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize, serde::Deserialize)]
pub enum ModelTier {
    /// Highest capability, highest cost (e.g. GPT-4o, Claude Opus).
    Premium = 3,
    /// Mid-tier capability and cost (e.g. GPT-4o-mini, Claude Sonnet).
    Standard = 2,
    /// Low-cost, reduced capability (e.g. GPT-3.5, Claude Haiku).
    Economy = 1,
    /// Zero marginal cost local model (e.g. Ollama, llama.cpp).
    Local = 0,
}

impl ModelTier {
    /// Return the next cheaper tier, or None if already at the lowest.
    pub fn downgrade(&self) -> Option<ModelTier> {
        match self {
            ModelTier::Premium => Some(ModelTier::Standard),
            ModelTier::Standard => Some(ModelTier::Economy),
            ModelTier::Economy => Some(ModelTier::Local),
            ModelTier::Local => None,
        }
    }

    /// Approximate cost per 1K output tokens in USD.
    pub fn cost_per_1k_output(&self) -> f64 {
        match self {
            ModelTier::Premium => 0.075,
            ModelTier::Standard => 0.015,
            ModelTier::Economy => 0.00125,
            ModelTier::Local => 0.0,
        }
    }

    /// Approximate cost per 1K input tokens in USD.
    pub fn cost_per_1k_input(&self) -> f64 {
        match self {
            ModelTier::Premium => 0.015,
            ModelTier::Standard => 0.003,
            ModelTier::Economy => 0.00025,
            ModelTier::Local => 0.0,
        }
    }

    /// Estimate cost for a request.
    pub fn estimate_cost(&self, input_tokens: u32, output_tokens: u32) -> f64 {
        (input_tokens as f64 / 1000.0) * self.cost_per_1k_input()
            + (output_tokens as f64 / 1000.0) * self.cost_per_1k_output()
    }
}

/// Automatic model downgrade cascade based on budget pressure thresholds.
#[derive(Debug, Clone)]
pub struct ModelCascade {
    /// Current tier.
    current: ModelTier,
    /// Downgrade when remaining budget falls below this fraction (0.0–1.0).
    pressure_threshold: f64,
    /// Total budget used for percentage calculation.
    total_budget: f64,
}

impl ModelCascade {
    /// Create a cascade starting at `tier` with the given pressure threshold.
    pub fn new(tier: ModelTier, pressure_threshold: f64, total_budget: f64) -> Result<Self, crate::error::BudgetError> {
        if !(0.0..=1.0).contains(&pressure_threshold) {
            return Err(crate::error::BudgetError::InvalidConfig(
                format!("pressure_threshold must be in [0.0, 1.0], got {pressure_threshold}"),
            ));
        }
        if total_budget <= 0.0 {
            return Err(crate::error::BudgetError::InvalidConfig(
                "total_budget must be positive".into(),
            ));
        }
        Ok(Self { current: tier, pressure_threshold, total_budget })
    }

    /// Current model tier.
    pub fn current(&self) -> ModelTier {
        self.current
    }

    /// Evaluate remaining budget and downgrade if under pressure.
    /// Returns Some(new_tier) if a downgrade occurred, None otherwise.
    pub fn evaluate(&mut self, remaining: f64) -> Option<ModelTier> {
        let fraction_remaining = remaining / self.total_budget;
        if fraction_remaining <= self.pressure_threshold {
            if let Some(next) = self.current.downgrade() {
                let old = self.current;
                self.current = next;
                let _ = old; // consumed
                return Some(self.current);
            }
        }
        None
    }

    /// Force downgrade one tier. Returns the new tier, or None if already Local.
    pub fn force_downgrade(&mut self) -> Option<ModelTier> {
        if let Some(next) = self.current.downgrade() {
            self.current = next;
            Some(self.current)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_tier_ordering() {
        assert!(ModelTier::Premium > ModelTier::Standard);
        assert!(ModelTier::Standard > ModelTier::Economy);
        assert!(ModelTier::Economy > ModelTier::Local);
    }

    #[test]
    fn test_downgrade_premium_to_standard() {
        assert_eq!(ModelTier::Premium.downgrade(), Some(ModelTier::Standard));
    }

    #[test]
    fn test_downgrade_standard_to_economy() {
        assert_eq!(ModelTier::Standard.downgrade(), Some(ModelTier::Economy));
    }

    #[test]
    fn test_downgrade_economy_to_local() {
        assert_eq!(ModelTier::Economy.downgrade(), Some(ModelTier::Local));
    }

    #[test]
    fn test_downgrade_local_returns_none() {
        assert_eq!(ModelTier::Local.downgrade(), None);
    }

    #[test]
    fn test_cost_per_1k_decreases_by_tier() {
        assert!(ModelTier::Premium.cost_per_1k_output() > ModelTier::Standard.cost_per_1k_output());
        assert!(ModelTier::Standard.cost_per_1k_output() > ModelTier::Economy.cost_per_1k_output());
        assert_eq!(ModelTier::Local.cost_per_1k_output(), 0.0);
    }

    #[test]
    fn test_estimate_cost_local_is_zero() {
        assert_eq!(ModelTier::Local.estimate_cost(10000, 5000), 0.0);
    }

    #[test]
    fn test_estimate_cost_premium_nonzero() {
        let cost = ModelTier::Premium.estimate_cost(1000, 1000);
        assert!(cost > 0.0);
    }

    #[test]
    fn test_cascade_new_invalid_threshold() {
        let result = ModelCascade::new(ModelTier::Premium, 1.5, 100.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_cascade_new_invalid_budget() {
        let result = ModelCascade::new(ModelTier::Premium, 0.2, 0.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_cascade_evaluate_triggers_downgrade_when_below_threshold() {
        let mut cascade = ModelCascade::new(ModelTier::Premium, 0.2, 100.0).expect("new");
        // Remaining = 15.0 / 100.0 = 0.15 < 0.20 threshold
        let result = cascade.evaluate(15.0);
        assert_eq!(result, Some(ModelTier::Standard));
        assert_eq!(cascade.current(), ModelTier::Standard);
    }

    #[test]
    fn test_cascade_evaluate_no_downgrade_when_above_threshold() {
        let mut cascade = ModelCascade::new(ModelTier::Premium, 0.2, 100.0).expect("new");
        // Remaining = 50.0 / 100.0 = 0.50 > 0.20 threshold
        let result = cascade.evaluate(50.0);
        assert_eq!(result, None);
        assert_eq!(cascade.current(), ModelTier::Premium);
    }

    #[test]
    fn test_cascade_local_cannot_downgrade() {
        let mut cascade = ModelCascade::new(ModelTier::Local, 0.5, 100.0).expect("new");
        let result = cascade.evaluate(10.0); // 10% remaining < 50% threshold
        assert_eq!(result, None); // no tier below Local
    }

    #[test]
    fn test_cascade_force_downgrade_chain() {
        let mut cascade = ModelCascade::new(ModelTier::Premium, 0.1, 100.0).expect("new");
        assert_eq!(cascade.force_downgrade(), Some(ModelTier::Standard));
        assert_eq!(cascade.force_downgrade(), Some(ModelTier::Economy));
        assert_eq!(cascade.force_downgrade(), Some(ModelTier::Local));
        assert_eq!(cascade.force_downgrade(), None);
    }

    #[test]
    fn test_model_tier_serde_roundtrip() {
        let tier = ModelTier::Standard;
        let json = serde_json::to_string(&tier).expect("serialize");
        let restored: ModelTier = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(restored, tier);
    }
}
