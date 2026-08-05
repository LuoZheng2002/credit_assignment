use std::{fs, path::Path};

use credit_assignment::chunked_judging::{
    JudgmentCacheRecord, cache_chunk_index, cache_chunk_path, judgment_cache_key, load_cache_chunk,
    rewrite_cache_chunk,
};

#[test]
fn cache_chunk_index_uses_flat_id_ranges() {
    assert_eq!(cache_chunk_index(0, 1000), 0);
    assert_eq!(cache_chunk_index(999, 1000), 0);
    assert_eq!(cache_chunk_index(1000, 1000), 1);
    assert_eq!(cache_chunk_index(2501, 1000), 2);
}

#[test]
fn cache_chunk_path_is_split_scoped() {
    let path = cache_chunk_path("/tmp/cache", "Training", 2501, 1000);
    assert_eq!(
        path,
        Path::new("/tmp/cache")
            .join("Training")
            .join("judgment_cache_chunk_00000002.jsonl")
    );
}

#[test]
fn cache_chunk_rewrite_preserves_one_record_per_key() {
    let temp_dir = tempfile::tempdir().unwrap();
    let path = temp_dir
        .path()
        .join("Training")
        .join("judgment_cache_chunk_00000000.jsonl");
    fs::create_dir_all(path.parent().unwrap()).unwrap();

    let key = judgment_cache_key("v1", "Training", 7, "42");
    let first = JudgmentCacheRecord {
        key: key.clone(),
        correct_answer: "41".to_string(),
        is_correct: false,
        decision_phase: "phase1_unanimous".to_string(),
        judge_outputs: Vec::new(),
        updated_unix_secs: 1,
    };
    let second = JudgmentCacheRecord {
        key: key.clone(),
        correct_answer: "42".to_string(),
        is_correct: true,
        decision_phase: "phase2_agreement".to_string(),
        judge_outputs: Vec::new(),
        updated_unix_secs: 2,
    };

    let mut records = std::collections::BTreeMap::new();
    records.insert(first.key.clone(), first);
    records.insert(second.key.clone(), second);
    rewrite_cache_chunk(&path, &records).unwrap();

    let loaded = load_cache_chunk(&path).unwrap();
    assert_eq!(loaded.len(), 1);
    let record = loaded.get(&key).unwrap();
    assert!(record.is_correct);
    assert_eq!(record.correct_answer, "42");
}
