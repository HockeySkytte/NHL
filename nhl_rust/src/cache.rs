//! In-process cache primitives mirroring the Flask module-level caches.
//!
//! Flask stores `(timestamp, payload)` tuples and checks ages manually
//! (`_cache_get` / `_cache_prune_ttl_and_size` / `_dict_set_bounded`). moka
//! provides native per-entry TTL and capacity eviction with the same
//! observable semantics ("stale entries are never served").

use std::hash::Hash;
use std::io::Write;
use std::time::Duration;

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

/// Serialized byte length of a serde_json value — a cheap memory proxy. Used
/// as the weigher for the large `Value` caches (Value working-set memory is a
/// small multiple of the serialized length, so capping weight bounds RAM).
pub fn json_value_weight(v: &serde_json::Value) -> u32 {
    let mut n = 0usize;
    let mut sink = CountingSink { n: &mut n };
    let _ = serde_json::to_writer(&mut sink, v);
    n.clamp(1, u32::MAX as usize) as u32
}

struct CountingSink<'a> {
    n: &'a mut usize,
}

impl Write for CountingSink<'_> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        *self.n += buf.len();
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

/// Deterministic single-slot cache: keeps at most ONE entry and replaces it
/// synchronously on every insert. Used for the huge line-tool datasets where a
/// second retained team would blow the memory budget — moka's capacity eviction
/// is lazy/asynchronous, which lets multiple giant entries coexist temporarily,
/// so it cannot be relied on here.
pub struct SingleSlot<K, V> {
    inner: std::sync::Mutex<Option<(K, V)>>,
}

impl<K, V> SingleSlot<K, V> {
    pub fn new() -> Self {
        Self {
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
        let guard = self.inner.lock().ok()?;
        match &*guard {
            Some((k, v)) if k == key => Some(v.clone()),
            _ => None,
        }
    }

    pub fn insert(&self, key: K, value: V) {
        if let Ok(mut guard) = self.inner.lock() {
            *guard = Some((key, value));
        }
    }

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
}
