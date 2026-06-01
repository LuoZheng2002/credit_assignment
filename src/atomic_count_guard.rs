use std::sync::{Arc, atomic::AtomicUsize};

use research_utility::log_message::log_key_value_pair;




pub struct AtomicCountGuard {
    count: Arc<AtomicUsize>,
    key: String,
}

impl AtomicCountGuard {
    pub fn new(count: Arc<AtomicUsize>, key: impl Into<String>) -> Self {
        let key = key.into();
        let new_count = count.fetch_add(1, std::sync::atomic::Ordering::SeqCst) + 1;
        log_key_value_pair(&key, new_count.to_string());
        Self { count, key }
    }

    pub fn try_new_with_max(
        count: Arc<AtomicUsize>,
        key: impl Into<String>,
        max_count: usize,
    ) -> Option<Self> {
        let key = key.into();
        let old_count = count
            .fetch_update(
                std::sync::atomic::Ordering::SeqCst,
                std::sync::atomic::Ordering::SeqCst,
                |current| {
                    if current < max_count {
                        Some(current + 1)
                    } else {
                        None
                    }
                },
            )
            .ok()?;
        log_key_value_pair(&key, (old_count + 1).to_string());
        Some(Self { count, key })
    }
}
impl Drop for AtomicCountGuard {
    fn drop(&mut self) {
        let new_count = self.count.fetch_sub(1, std::sync::atomic::Ordering::SeqCst) - 1;
        log_key_value_pair(&self.key, new_count.to_string());
    }
}
