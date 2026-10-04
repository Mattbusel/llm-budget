//! Budget persistence: save and restore budget state across runs, as JSON files.

use crate::budget::BudgetSnapshot;
use crate::error::BudgetError;
use std::path::PathBuf;
use tokio::sync::RwLock;
use std::collections::HashMap;
use std::sync::Arc;

/// Serializable budget state for a session.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
pub struct PersistedBudgetState {
    /// Session name the state was saved under.
    pub session_id: String,
    /// One snapshot per agent.
    pub snapshots: Vec<BudgetSnapshot>,
    /// Unix time in milliseconds of the save.
    pub saved_at_ms: u64,
}

/// Reads and writes `<session>.budget.json` files in one directory.
pub struct BudgetStore {
    base_path: PathBuf,
    cache: Arc<RwLock<HashMap<String, PersistedBudgetState>>>,
}

impl BudgetStore {
    /// Store files under `base_path` (created on first save).
    pub fn new(base_path: impl Into<PathBuf>) -> Self {
        Self {
            base_path: base_path.into(),
            cache: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Save budget snapshots for a session.
    ///
    /// Session ids may contain letters, digits, `-`, `_` and `.` only, so an id
    /// can never point outside the store directory.
    ///
    /// # Errors
    /// [`BudgetError::PersistenceError`] for an invalid id or an I/O failure.
    pub async fn save(
        &self,
        session_id: &str,
        snapshots: Vec<BudgetSnapshot>,
    ) -> Result<(), BudgetError> {
        let path = self.session_path(session_id)?;
        let state = PersistedBudgetState {
            session_id: session_id.to_string(),
            snapshots,
            saved_at_ms: now_ms(),
        };

        if let Some(parent) = path.parent() {
            tokio::fs::create_dir_all(parent).await
                .map_err(|e| BudgetError::PersistenceError(e.to_string()))?;
        }
        let json = serde_json::to_string_pretty(&state)
            .map_err(|e| BudgetError::PersistenceError(e.to_string()))?;
        // Write to a temporary file and rename it, so a crash mid-write never
        // leaves a truncated JSON file behind.
        let tmp = path.with_extension("json.tmp");
        tokio::fs::write(&tmp, json).await
            .map_err(|e| BudgetError::PersistenceError(e.to_string()))?;
        tokio::fs::rename(&tmp, &path).await
            .map_err(|e| BudgetError::PersistenceError(e.to_string()))?;

        self.cache.write().await.insert(session_id.to_string(), state);
        Ok(())
    }

    /// Load budget state for a session, checking the in-memory cache first.
    ///
    /// # Errors
    /// [`BudgetError::PersistenceError`] if the id is invalid, the file is
    /// missing or it does not parse.
    pub async fn load(&self, session_id: &str) -> Result<PersistedBudgetState, BudgetError> {
        let path = self.session_path(session_id)?;
        if let Some(state) = self.cache.read().await.get(session_id) {
            return Ok(state.clone());
        }

        let data = tokio::fs::read_to_string(&path).await
            .map_err(|_| BudgetError::PersistenceError(
                format!("session '{}' not found on disk", session_id)
            ))?;
        let state: PersistedBudgetState = serde_json::from_str(&data)
            .map_err(|e| BudgetError::PersistenceError(e.to_string()))?;

        self.cache.write().await.insert(session_id.to_string(), state.clone());
        Ok(state)
    }

    /// Delete persisted state for a session. Deleting a missing session is not an error.
    ///
    /// # Errors
    /// [`BudgetError::PersistenceError`] for an invalid id or an I/O failure.
    pub async fn delete(&self, session_id: &str) -> Result<(), BudgetError> {
        let path = self.session_path(session_id)?;
        self.cache.write().await.remove(session_id);
        match tokio::fs::remove_file(&path).await {
            Ok(()) => Ok(()),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(e) => Err(BudgetError::PersistenceError(e.to_string())),
        }
    }

    /// List all session IDs that have persisted state.
    ///
    /// # Errors
    /// [`BudgetError::PersistenceError`] if the directory cannot be read.
    pub async fn list_sessions(&self) -> Result<Vec<String>, BudgetError> {
        let mut sessions = Vec::new();
        let mut dir = tokio::fs::read_dir(&self.base_path).await
            .map_err(|e| BudgetError::PersistenceError(e.to_string()))?;
        while let Some(entry) = dir.next_entry().await
            .map_err(|e| BudgetError::PersistenceError(e.to_string()))?
        {
            let name = entry.file_name();
            let name_str = name.to_string_lossy();
            if name_str.ends_with(".budget.json") {
                let session_id = name_str.trim_end_matches(".budget.json").to_string();
                sessions.push(session_id);
            }
        }
        Ok(sessions)
    }

    fn session_path(&self, session_id: &str) -> Result<PathBuf, BudgetError> {
        let valid = !session_id.is_empty()
            && session_id.len() <= 200
            && !session_id.starts_with('.')
            && session_id
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'));
        if !valid {
            return Err(BudgetError::PersistenceError(format!(
                "invalid session id '{session_id}': use letters, digits, '-', '_' or '.'"
            )));
        }
        Ok(self.base_path.join(format!("{session_id}.budget.json")))
    }
}

fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::ModelTier;
    use tempfile::TempDir;

    fn make_snapshot(agent_id: &str, spent: f64, limit: f64) -> BudgetSnapshot {
        BudgetSnapshot {
            agent_id: agent_id.into(),
            spent_usd: spent,
            limit_usd: limit,
            remaining_usd: limit - spent,
            utilization: spent / limit,
            is_killed: false,
            model_tier: ModelTier::Standard,
            timestamp_ms: 0,
        }
    }

    #[tokio::test]
    async fn test_budget_store_save_and_load() {
        let dir = TempDir::new().expect("tempdir");
        let store = BudgetStore::new(dir.path());
        let snapshots = vec![make_snapshot("a1", 1.0, 10.0)];
        store.save("session-1", snapshots).await.expect("save");
        let loaded = store.load("session-1").await.expect("load");
        assert_eq!(loaded.session_id, "session-1");
        assert_eq!(loaded.snapshots.len(), 1);
        assert_eq!(loaded.snapshots[0].agent_id, "a1");
    }

    #[tokio::test]
    async fn test_budget_store_load_missing_returns_error() {
        let dir = TempDir::new().expect("tempdir");
        let store = BudgetStore::new(dir.path());
        let result = store.load("nonexistent").await;
        assert!(matches!(result, Err(BudgetError::PersistenceError(_))));
    }

    #[tokio::test]
    async fn test_budget_store_cache_hit_avoids_disk_read() {
        let dir = TempDir::new().expect("tempdir");
        let store = BudgetStore::new(dir.path());
        let snapshots = vec![make_snapshot("a1", 2.0, 10.0)];
        store.save("s1", snapshots).await.expect("save");
        // Second load should come from cache
        let loaded = store.load("s1").await.expect("load from cache");
        assert_eq!(loaded.snapshots[0].spent_usd, 2.0);
    }

    #[tokio::test]
    async fn test_budget_store_delete_removes_session() {
        let dir = TempDir::new().expect("tempdir");
        let store = BudgetStore::new(dir.path());
        store.save("s1", vec![make_snapshot("a1", 1.0, 5.0)]).await.expect("save");
        store.delete("s1").await.expect("delete");
        let result = store.load("s1").await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_budget_store_list_sessions() {
        let dir = TempDir::new().expect("tempdir");
        let store = BudgetStore::new(dir.path());
        store.save("sess-a", vec![make_snapshot("x", 1.0, 5.0)]).await.expect("save a");
        store.save("sess-b", vec![make_snapshot("y", 2.0, 5.0)]).await.expect("save b");
        let mut sessions = store.list_sessions().await.expect("list");
        sessions.sort();
        assert_eq!(sessions, vec!["sess-a", "sess-b"]);
    }

    #[tokio::test]
    async fn test_budget_store_multiple_snapshots_per_session() {
        let dir = TempDir::new().expect("tempdir");
        let store = BudgetStore::new(dir.path());
        let snapshots = vec![
            make_snapshot("a1", 1.0, 10.0),
            make_snapshot("a2", 3.0, 10.0),
            make_snapshot("a3", 0.5, 10.0),
        ];
        store.save("fleet-session", snapshots).await.expect("save");
        let loaded = store.load("fleet-session").await.expect("load");
        assert_eq!(loaded.snapshots.len(), 3);
    }

    #[tokio::test]
    async fn test_budget_store_overwrite_session() {
        let dir = TempDir::new().expect("tempdir");
        let store = BudgetStore::new(dir.path());
        store.save("s1", vec![make_snapshot("a1", 1.0, 10.0)]).await.expect("first");
        store.save("s1", vec![make_snapshot("a1", 5.0, 10.0)]).await.expect("second");
        // Evict cache to force disk read
        store.cache.write().await.clear();
        let loaded = store.load("s1").await.expect("load after overwrite");
        assert_eq!(loaded.snapshots[0].spent_usd, 5.0);
    }

    #[tokio::test]
    async fn test_session_id_cannot_escape_store_dir() {
        // Before 0.2.0 "../outside" was joined onto the base path as-is and the
        // file was written one directory up.
        let dir = TempDir::new().expect("tempdir");
        let inner = dir.path().join("store");
        let store = BudgetStore::new(&inner);
        for bad in ["../outside", "a/b", "a\\b", "", "..", ".hidden"] {
            assert!(store.save(bad, vec![]).await.is_err(), "accepted {bad:?}");
            assert!(store.load(bad).await.is_err());
            assert!(store.delete(bad).await.is_err());
        }
        assert!(!dir.path().join("outside.budget.json").exists());
        store.save("ok-id_1.v2", vec![]).await.expect("valid id");
    }

    #[tokio::test]
    async fn test_delete_missing_session_is_ok() {
        let dir = TempDir::new().expect("tempdir");
        let store = BudgetStore::new(dir.path());
        store.delete("never-saved").await.expect("delete missing");
    }
}
