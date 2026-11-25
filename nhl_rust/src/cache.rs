//! In-process cache primitives mirroring the Flask module-level caches.
//!
//! Flask stores `(timestamp, payload)` tuples and checks ages manually
//! (`_cache_get` / `_cache_prune_ttl_and_size` / `_dict_set_bounded`). moka
//! provides native per-entry TTL and capacity eviction with the same
//! observable semantics ("stale entries are never served").

use std::hash::Hash;
use std::time::{Duration, Instant};

use moka::sync::Cache;

/// Reads a `*_CACHE_TTL_SECONDS` env var with a default, clamped >= 1.
/// Mirrors the Flask env conventions.
pub fn env_ttl(name: &str, default: u64) -> Duration {
    let secs = std::env::var(name)
        .ok()
        .and_then(|v| v.trim().parse::<u64>().ok())
        .unwrap_or(default);
    Duration::from_secs(secs.max(1))
}

/// Reads a `*_CACHE_MAX_ITEMS` env var with a default, clamped >= 1.
pub fn env_max(name: &str, default: u64) -> u64 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.trim().parse::<u64>().ok())
        .unwrap_or(default)
        .max(1)
}

/// TTL + max-capacity cache — replaces the Python `(ts, payload)` tuple caches.
pub struct TtlCache<K, V> {
    inner: Cache<K, V>,
}

impl<K, V> TtlCache<K, V>
where
    K: Clone + Eq + Hash + Send + Sync + 'static,
    V: Clone + Send + Sync + 'static,
{
    pub fn new(ttl: Duration, max_items: u64) -> Self {
        let builder = Cache::builder()
            .time_to_live(ttl)
            .max_capacity(max_items.max(1));
        Self {
            inner: builder.build(),
        }
    }

    /// TTL + weight-bounded cache: `max_weight` is the total serialized-byte
    /// budget for the cache (eviction is by weight under pressure). This keeps
    /// memory bounded even when entries are huge (e.g. full-season aggregates).
    pub fn new_weighted(
        ttl: Duration,
        max_weight: u64,
        weigher: impl Fn(&V) -> u32 + Send + Sync + 'static,
    ) -> Self {
        let builder = Cache::builder()
            .time_to_live(ttl)
            .weigher(move |_k, v: &V| weigher(v).max(1))
            .max_capacity(max_weight.max(1));
        Self {
            inner: builder.build(),
        }
    }

    pub fn get(&self, key: &K) -> Option<V> {
        self.inner.get(key)
    }

    pub fn insert(&self, key: K, value: V) {
        self.inner.insert(key, value);
    }

    pub fn invalidate(&self, key: &K) {
        self.inner.invalidate(key);
    }

    pub fn contains_key(&self, key: &K) -> bool {
        self.inner.contains_key(key)
    }
}

/// Fast approximate byte length of a `serde_json::Value` without running a
/// full JSON serializer (avoids float-to-decimal formatting on huge trees).
/// Used as the weigher for the large `Value` caches.
pub fn json_value_weight(v: &serde_json::Value) -> u32 {
    approx_json_weight(v).clamp(1, u32::MAX as usize) as u32
}

/// Fast approximate byte length of a `&[Value]` row array.
pub fn json_rows_weight(rows: &[serde_json::Value]) -> u32 {
    let mut n = 2usize;
    for r in rows {
        n = n.saturating_add(approx_json_weight(r)).saturating_add(1);
    }
    n.clamp(1, u32::MAX as usize) as u32
}

fn approx_json_weight(v: &serde_json::Value) -> usize {
    match v {
        serde_json::Value::Null => 4,
        serde_json::Value::Bool(_) => 5,
        serde_json::Value::Number(_) => 10,
        serde_json::Value::String(s) => s.len().saturating_add(2),
        serde_json::Value::Array(arr) => {
            let mut n = 2usize;
            for item in arr {
                n = n.saturating_add(approx_json_weight(item)).saturating_add(1);
            }
            n
        }
        serde_json::Value::Object(map) => {
            let mut n = 2usize;
            for (k, val) in map {
                n = n
                    .saturating_add(k.len())
                    .saturating_add(4)
                    .saturating_add(approx_json_weight(val));
            }
            n
        }
    }
}

/// Deterministic single-slot cache: keeps at most ONE entry and replaces it
/// synchronously on every insert, with an optional TTL so stale entries expire.
/// Used with `Arc<V>` for the huge line-tool datasets so hits are an O(1)
/// ref-count bump instead of a deep clone under a mutex.
pub struct SingleSlot<K, V> {
    ttl: Option<Duration>,
    inner: std::sync::Mutex<Option<(K, V, Instant)>>,
}

impl<K, V> SingleSlot<K, V> {
    #[allow(dead_code)]
    pub fn new() -> Self {
        Self {
            ttl: None,
            inner: std::sync::Mutex::new(None),
        }
    }

    pub fn new_with_ttl(ttl: Duration) -> Self {
        Self {
            ttl: Some(ttl),
            inner: std::sync::Mutex::new(None),
        }
    }
}

impl<K, V> Default for SingleSlot<K, V> {
    fn default() -> Self {
        Self::new()
    }
}

impl<K, V> SingleSlot<K, V>
where
    K: Clone + Eq,
    V: Clone,
{
    pub fn get(&self, key: &K) -> Option<V> {
        let mut guard = self.inner.lock().ok()?;
        let expired = match &*guard {
            Some((k, _, ts)) if k == key => self.ttl.is_some_and(|ttl| ts.elapsed() >= ttl),
            _ => return None,
        };
        if expired {
            *guard = None;
            return None;
        }
        guard.as_ref().map(|(_, v, _)| v.clone())
    }

    pub fn insert(&self, key: K, value: V) {
        if let Ok(mut guard) = self.inner.lock() {
            *guard = Some((key, value, Instant::now()));
        }
    }

    #[allow(dead_code)]
    pub fn invalidate(&self, _key: &K) {
        if let Ok(mut guard) = self.inner.lock() {
            *guard = None;
        }
    }
}

/// Capacity-bounded cache without TTL — replaces `_dict_set_bounded`
/// (insertion-order LRU approximation).
pub struct BoundedCache<K, V> {
    inner: Cache<K, V>,
}

impl<K, V> BoundedCache<K, V>
where
    K: Clone + Eq + Hash + Send + Sync + 'static,
    V: Clone + Send + Sync + 'static,
{
    pub fn new(max_items: u64) -> Self {
        Self {
            inner: Cache::builder().max_capacity(max_items.max(1)).build(),
        }
    }

    pub fn get(&self, key: &K) -> Option<V> {
        self.inner.get(key)
    }

    pub fn insert(&self, key: K, value: V) {
        self.inner.insert(key, value);
    }

    /// Approximate current entry count (exact for the sync cache without
    /// eviction listeners).
    pub fn len(&self) -> u64 {
        self.inner.entry_count()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn invalidate(&self, key: &K) {
        self.inner.invalidate(key);
    }
}

/// Request coalescing ("singleflight") for expensive loads: concurrent calls
/// with the same key share ONE load instead of each doing the full work
/// (e.g. N users triggering N full-table Supabase reads on a cache miss).
///
/// The owner holds a per-key async mutex for the duration of the load;
/// waiters queue on it, then re-run the closure (which re-checks the cache
/// first), so a finished load is never duplicated and a dropped/cancelled
/// owner never deadlocks the key.
pub struct InFlight {
    map: std::sync::Arc<
        std::sync::Mutex<std::collections::HashMap<String, std::sync::Arc<tokio::sync::Mutex<()>>>>,
    >,
}

impl InFlight {
    pub fn new() -> Self {
        Self {
            map: std::sync::Arc::new(std::sync::Mutex::new(std::collections::HashMap::new())),
        }
    }

    pub async fn run<T, F, Fut>(&self, key: &str, f: F) -> T
    where
        F: FnOnce() -> Fut + Send,
        Fut: std::future::Future<Output = T> + Send,
    {
        loop {
            let slot = {
                let mut map = self.map.lock().unwrap_or_else(|p| p.into_inner());
                match map.get(key) {
                    Some(m) => (m.clone(), false),
                    None => {
                        let m = std::sync::Arc::new(tokio::sync::Mutex::new(()));
                        map.insert(key.to_string(), m.clone());
                        (m, true)
                    }
                }
            };
            if !slot.1 {
                // Another task is loading; wait for it to finish, then loop
                // back so the closure re-checks the (now warm) cache.
                let _guard = slot.0.lock().await;
                continue;
            }
            // We own this key: run the load while holding the lock so other
            // tasks on the same key queue behind us. An RAII guard ensures the
            // map entry is removed even if this future is cancelled or panics
            // (preventing a waiter livelock).
            let mutex = slot.0;
            let _guard = mutex.lock().await;
            let _cleanup = InFlightCleanup {
                map: &self.map,
                key,
            };
            return f().await;
        }
    }
}

struct InFlightCleanup<'a> {
    map: &'a std::sync::Mutex<std::collections::HashMap<String, std::sync::Arc<tokio::sync::Mutex<()>>>>,
    key: &'a str,
}

impl Drop for InFlightCleanup<'_> {
    fn drop(&mut self) {
        if let Ok(mut map) = self.map.lock() {
            map.remove(self.key);
        }
    }
}

impl Default for InFlight {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bounded_cache_respects_capacity() {
        let c = BoundedCache::new(2);
        c.insert(1, "a");
        c.insert(2, "b");
        c.insert(3, "c");
        // moka eviction is best-effort on timing; capacity is the contract.
        assert!(c.len() <= 2);
        // The most recent insert is always retrievable.
        assert_eq!(c.get(&3), Some("c"));
    }

    #[test]
    fn ttl_cache_expires_entries() {
        let c = TtlCache::new(Duration::from_millis(10), 8);
        c.insert("k", 1usize);
        assert_eq!(c.get(&"k"), Some(1));
        std::thread::sleep(Duration::from_millis(40));
        assert!(c.get(&"k").is_none());
    }

    #[test]
    fn bounded_cache_invalidate() {
        let c = BoundedCache::new(4);
        c.insert("k", 1usize);
        c.invalidate(&"k");
        assert!(c.get(&"k").is_none());
    }

    #[test]
    fn single_slot_ttl_expires() {
        let s = SingleSlot::new_with_ttl(Duration::from_millis(15));
        s.insert("k".to_string(), 42usize);
        assert_eq!(s.get(&"k".to_string()), Some(42));
        std::thread::sleep(Duration::from_millis(30));
        assert_eq!(s.get(&"k".to_string()), None);
    }

    #[tokio::test]
    async fn inflight_recovers_after_owner_cancellation() {
        let inflight = std::sync::Arc::new(InFlight::new());
        let inf1 = inflight.clone();
        let handle = tokio::spawn(async move {
            inf1.run("k", || async {
                tokio::time::sleep(Duration::from_secs(60)).await;
                1usize
            })
            .await
        });
        tokio::time::sleep(Duration::from_millis(20)).await;
        handle.abort();
        let _ = handle.await;

        // Next caller must complete cleanly instead of livelocking.
        let res = tokio::time::timeout(
            Duration::from_secs(2),
            inflight.run("k", || async { 99usize }),
        )
        .await
        .expect("inflight must not livelock after cancellation");
        assert_eq!(res, 99);
    }
}
