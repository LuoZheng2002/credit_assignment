use indexmap::IndexMap;
use serde::{Serialize, de::DeserializeOwned};
use std::collections::HashSet;
use std::fs::{File, OpenOptions};
use std::future::Future;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};
use tokio::sync::{Semaphore, mpsc};

pub trait HasId {
    fn id(&self) -> usize;
}

pub fn read_json_lines<T: DeserializeOwned>(path: impl AsRef<Path>) -> Result<Vec<T>, String> {
    let file = File::open(path.as_ref())
        .map_err(|e| format!("Cannot open file {}: {}", path.as_ref().display(), e))?;
    let reader = BufReader::new(file);
    let mut results = Vec::new();
    for line in reader.lines() {
        let line = line.map_err(|e| e.to_string())?;
        let item: T = serde_json::from_str(&line)
            .map_err(|e| format!("Failed to parse line: {}\nError: {}", line, e))?;
        results.push(item);
    }
    Ok(results)
}

pub fn read_json_lines_indexed<T: DeserializeOwned + HasId>(
    path: impl AsRef<Path>,
) -> Result<IndexMap<usize, T>, String> {
    let file = File::open(path.as_ref())
        .map_err(|e| format!("Cannot open file {}: {}", path.as_ref().display(), e))?;
    let reader = BufReader::new(file);
    let mut results = IndexMap::new();
    for line in reader.lines() {
        let line = line.map_err(|e| e.to_string())?;
        let item: T = serde_json::from_str(&line)
            .map_err(|e| format!("Failed to parse line: {}\nError: {}", line, e))?;
        results.insert(item.id(), item);
    }
    Ok(results)
}

pub fn read_json_lines_indexed_erased(
    path: impl AsRef<Path>,
) -> Result<IndexMap<usize, serde_json::Value>, String> {
    let file = File::open(path.as_ref())
        .map_err(|e| format!("Cannot open file {}: {}", path.as_ref().display(), e))?;
    let reader = BufReader::new(file);
    let mut results = IndexMap::new();
    for line in reader.lines() {
        let line = line.map_err(|e| e.to_string())?;
        let item: serde_json::Value = serde_json::from_str(&line).map_err(|e| e.to_string())?;
        // let id = item.get("id").expect("Missing id field").as_u64().unwrap() as usize;
        let id = item
            .get("id")
            .ok_or_else(|| format!("Missing id field in item: {}", item))?
            .as_u64()
            .ok_or_else(|| format!("id field is not a number in item: {}", item))?
            as usize;
        results.insert(id, item);
    }
    Ok(results)
}

pub fn write_jsonl_file<T: Serialize>(
    file_path: impl AsRef<Path>,
    data: &[T],
) -> Result<(), String> {
    let file_path = file_path.as_ref();
    if let Some(parent) = file_path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    let mut file = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(file_path)
        .map_err(|e| e.to_string())?;
    for item in data {
        let line = serde_json::to_string(item).map_err(|e| e.to_string())?;
        writeln!(file, "{}", line).map_err(|e| e.to_string())?;
    }
    Ok(())
}

// currently parallel_process_jsonl first reads the input files to get the items without holding the file handles
// then the items are zipped.
// then we open an output file and persistently append to it
// in fact, we can first read it, and then append it
// then we remove processed zipped items based on the contents in the output file
// then we submit tasks and keep receiving processed items, and append to the file
// then we sort the file at the end.

// we want to modify the process so that the semaphore is in the for loop
//
// for the new task, we need to read from multiple files and append to multiple files.
// the rollout function is responsible for sending log appending and trajectory appending signals
// we need separate async tasks to read and execute the appending
// the main function needs to wait for any of the appending to finish before exiting

pub async fn parallel_process_jsonl<T, U, F, Z, Fut>(
    input_file_paths: &[impl AsRef<Path>],
    output_file_path: impl AsRef<Path>,
    zip_fn: Z,
    process_fn: F,
    max_tasks: usize,
) -> Result<(), String>
where
    T: DeserializeOwned + HasId + Send + 'static,
    U: Serialize + HasId + DeserializeOwned + Send + 'static,
    Z: Fn(&[&serde_json::Value]) -> T + Send + Sync + 'static,
    F: Fn(T) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = U> + Send + 'static,
{
    assert!(
        !input_file_paths.is_empty(),
        "At least one input file is required"
    );
    let mut items: Vec<IndexMap<usize, serde_json::Value>> = Vec::new();
    for input_file_path in input_file_paths {
        let file_items = read_json_lines_indexed_erased(input_file_path.as_ref())?;
        items.push(file_items);
    }
    // assert file_items have the same keys
    let first_keys: HashSet<usize> = items[0].keys().cloned().collect();
    for item_map in &items[1..] {
        let keys: HashSet<usize> = item_map.keys().cloned().collect();
        assert_eq!(keys, first_keys, "Input files have different keys");
    }
    let mut zipped_items = IndexMap::new();
    for id in first_keys {
        let item_values: Vec<&serde_json::Value> = items
            .iter()
            .map(|item_map| item_map.get(&id).unwrap())
            .collect();
        let zipped_item = zip_fn(&item_values);
        zipped_items.insert(id, zipped_item);
    }
    drop(items);

    let output_path = output_file_path.as_ref();
    if let Some(parent) = output_path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    let mut output_file = OpenOptions::new()
        .create(true)
        .read(true)
        .append(true)
        .open(output_path)
        .map_err(|e| e.to_string())?;
    let mut results = read_json_lines_indexed::<U>(output_path)?;

    let processed_ids: Vec<usize> = results.keys().cloned().collect();
    println!(
        "Already processed {} items, skipping them",
        processed_ids.len()
    );
    // for id in processed_ids {
    //     zipped_items.shift_remove(&id);
    // }
    zipped_items.retain(|id, _| !processed_ids.contains(id));
    println!("Processing {} items", zipped_items.len());

    let sem = Arc::new(Semaphore::new(max_tasks));
    let process_fn = Arc::new(process_fn);
    let (tx, mut rx) = mpsc::channel::<U>(max_tasks);
    let finished_count = Arc::new(AtomicUsize::new(0));

    for (_, task) in zipped_items.into_iter() {
        let sem = sem.clone();
        let tx = tx.clone();
        let count = finished_count.clone();
        let process_fn = process_fn.clone();
        tokio::spawn(async move {
            let _permit = sem.acquire_owned().await.unwrap();
            let answer = (&*process_fn)(task).await;
            count.fetch_add(1, Ordering::SeqCst);
            println!("Processed {} items", count.load(Ordering::SeqCst));
            tx.send(answer).await.unwrap();
        });
    }
    drop(tx);
    while let Some(answer) = rx.recv().await {
        let serialized = serde_json::to_string(&answer).map_err(|e| e.to_string())?;
        writeln!(output_file, "{}", serialized).map_err(|e| e.to_string())?;
        results.insert(answer.id(), answer);
    }
    results.sort_keys();
    let results_vec: Vec<&U> = results.values().collect();
    drop(output_file);
    write_jsonl_file(output_path, &results_vec)?;
    Ok(())
}
