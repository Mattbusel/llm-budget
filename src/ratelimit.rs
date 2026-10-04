//! Tokens-per-minute and requests-per-minute limits (feature `ratelimit`).
//!
//! Providers cap how fast you may spend as well as how much: OpenAI and
//! Anthropic both publish per-minute token (TPM) and request (RPM) limits per
//! model. A dollar budget does not stop an agent from hitting those, and a
//! `429 Too Many Requests` storm wastes calls. [`TokenRateLimiter`] checks a
//! call against both limits before you make it, using the GCRA algorithm from
//! [governor](https://crates.io/crates/governor).
//!
//! ```
//! use llm_budget::ratelimit::TokenRateLimiter;
//!
//! // 30,000 tokens and 500 requests per minute, like a low OpenAI tier.
//! let limiter = TokenRateLimiter::per_minute(30_000, Some(500))?;
//! limiter.check(12_000)?;            // fine
//! limiter.check(12_000)?;            // fine, 24,000 used
//! let err = limiter.check(12_000).unwrap_err();
//! assert!(err.to_string().contains("rate limit"));
//! # Ok::<(), llm_budget::BudgetError>(())
//! ```

use crate::error::BudgetError;
use governor::clock::{Clock, MonotonicClock};
use governor::middleware::NoOpMiddleware;
use governor::state::{InMemoryState, NotKeyed};
use governor::{Quota, RateLimiter};
use std::num::NonZeroU32;

type Limiter<C> = RateLimiter<NotKeyed, InMemoryState, C, NoOpMiddleware<<C as Clock>::Instant>>;

/// A per-minute limit on tokens and, optionally, on requests.
///
/// Both limits refill continuously (a 60,000 TPM limit gives back 1,000 tokens
/// a second), and a full minute's worth may be used in one burst. Safe to share
/// between threads (`&self` methods, `Send + Sync`).
pub struct TokenRateLimiter<C: Clock = MonotonicClock> {
    tokens: Limiter<C>,
    requests: Option<Limiter<C>>,
    tokens_per_minute: u32,
    clock: C,
}

fn nz(n: u32, what: &str) -> Result<NonZeroU32, BudgetError> {
    NonZeroU32::new(n).ok_or_else(|| BudgetError::InvalidConfig(format!("{what} per minute must be at least 1")))
}

impl TokenRateLimiter<MonotonicClock> {
    /// Allow `tokens_per_minute` tokens and, if given, `requests_per_minute` calls.
    ///
    /// # Errors
    /// [`BudgetError::InvalidConfig`] if a limit is 0.
    pub fn per_minute(tokens_per_minute: u32, requests_per_minute: Option<u32>) -> Result<Self, BudgetError> {
        Self::with_clock(tokens_per_minute, requests_per_minute, MonotonicClock)
    }
}

impl<C: Clock + Clone> TokenRateLimiter<C> {
    /// Same as [`TokenRateLimiter::per_minute`] with your own clock (for tests,
    /// governor's `FakeRelativeClock`).
    ///
    /// # Errors
    /// [`BudgetError::InvalidConfig`] if a limit is 0.
    pub fn with_clock(tokens_per_minute: u32, requests_per_minute: Option<u32>, clock: C) -> Result<Self, BudgetError> {
        let tokens = RateLimiter::direct_with_clock(Quota::per_minute(nz(tokens_per_minute, "tokens")?), clock.clone());
        let requests = match requests_per_minute {
            Some(r) => Some(RateLimiter::direct_with_clock(Quota::per_minute(nz(r, "requests")?), clock.clone())),
            None => None,
        };
        Ok(Self { tokens, requests, tokens_per_minute, clock })
    }

    /// Take `tokens` tokens and one request from the limits if both have room.
    ///
    /// The request limit is checked first; if the token limit then refuses,
    /// the request slot stays used (the error is conservative, never lenient).
    ///
    /// # Errors
    /// [`BudgetError::RateLimited`] with how long to wait, or
    /// [`BudgetError::InvalidConfig`] if `tokens` is more than a whole minute's
    /// allowance and could never pass.
    pub fn check(&self, tokens: u32) -> Result<(), BudgetError> {
        if let Some(req) = &self.requests {
            if let Err(not_until) = req.check() {
                return Err(self.limited(not_until.wait_time_from(self.clock.now())));
            }
        }
        let Some(n) = NonZeroU32::new(tokens) else {
            return Ok(());
        };
        match self.tokens.check_n(n) {
            Ok(Ok(())) => Ok(()),
            Ok(Err(not_until)) => Err(self.limited(not_until.wait_time_from(self.clock.now()))),
            Err(_) => Err(BudgetError::InvalidConfig(format!(
                "a call of {tokens} tokens can never fit a limit of {} tokens per minute",
                self.tokens_per_minute
            ))),
        }
    }

    fn limited(&self, wait: std::time::Duration) -> BudgetError {
        // Round up so "retry after" is never too early.
        let ms = wait.as_micros().div_ceil(1000);
        BudgetError::RateLimited { retry_after_ms: u64::try_from(ms).unwrap_or(u64::MAX) }
    }

    /// The token limit per minute.
    pub fn tokens_per_minute(&self) -> u32 {
        self.tokens_per_minute
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use governor::clock::FakeRelativeClock;
    use std::time::Duration;

    #[test]
    fn refuses_past_the_token_limit_and_refills() {
        let clock = FakeRelativeClock::default();
        let l = TokenRateLimiter::with_clock(60_000, None, clock.clone()).unwrap();
        l.check(60_000).unwrap();
        match l.check(1_000) {
            Err(BudgetError::RateLimited { retry_after_ms }) => assert_eq!(retry_after_ms, 1_000),
            other => panic!("expected RateLimited, got {other:?}"),
        }
        clock.advance(Duration::from_secs(1)); // 1,000 tokens come back
        l.check(1_000).unwrap();
        assert!(l.check(1).is_err());
    }

    #[test]
    fn request_limit_applies_even_to_small_calls() {
        let clock = FakeRelativeClock::default();
        let l = TokenRateLimiter::with_clock(1_000_000, Some(2), clock.clone()).unwrap();
        l.check(1).unwrap();
        l.check(1).unwrap();
        match l.check(1) {
            Err(BudgetError::RateLimited { retry_after_ms }) => assert_eq!(retry_after_ms, 30_000),
            other => panic!("expected RateLimited, got {other:?}"),
        }
        clock.advance(Duration::from_secs(30));
        l.check(1).unwrap();
    }

    #[test]
    fn impossible_calls_and_bad_config() {
        let l = TokenRateLimiter::per_minute(100, None).unwrap();
        assert!(matches!(l.check(101), Err(BudgetError::InvalidConfig(_))));
        assert!(l.check(0).is_ok());
        assert!(TokenRateLimiter::per_minute(0, None).is_err());
        assert!(TokenRateLimiter::per_minute(10, Some(0)).is_err());
        assert_eq!(l.tokens_per_minute(), 100);
    }

    #[test]
    fn is_send_and_sync() {
        fn check<T: Send + Sync>() {}
        check::<TokenRateLimiter>();
    }
}
