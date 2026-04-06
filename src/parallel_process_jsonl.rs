use indexmap::IndexMap;
use serde::{Serialize, de::DeserializeOwned};
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

pub fn read_json_lines_indexed<T: DeserializeOwned + HasId>(
    file: &File,
) -> Result<IndexMap<usize, T>, String> {
    let reader = BufReader::new(file);
    let mut results = IndexMap::new();
    for line in reader.lines() {
        let line = line.map_err(|e| e.to_string())?;
        let item: T = serde_json::from_str(&line).map_err(|e| e.to_string())?;
        results.insert(item.id(), item);
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

pub async fn parallel_process_jsonl<T, U, F, Fut>(
    input_file_path: impl AsRef<Path>,
    output_file_path: impl AsRef<Path>,
    process_fn: F,
    max_tasks: usize,
) -> Result<(), String>
where
    T: DeserializeOwned + HasId + Send + 'static,
    U: Serialize + HasId + DeserializeOwned + Send + 'static,
    F: Fn(T) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = U> + Send + 'static,
{
    let input_file = File::open(input_file_path.as_ref()).map_err(|e| e.to_string())?;
    let mut items = read_json_lines_indexed::<T>(&input_file)?;

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
    let mut results = read_json_lines_indexed::<U>(&output_file)?;

    let processed_ids: Vec<usize> = results.keys().cloned().collect();
    println!(
        "Already processed {} items, skipping them",
        processed_ids.len()
    );
    for id in processed_ids {
        items.shift_remove(&id);
    }
    println!("Processing {} items", items.len());

    let sem = Arc::new(Semaphore::new(max_tasks));
    let process_fn = Arc::new(process_fn);
    let (tx, mut rx) = mpsc::channel::<U>(max_tasks);
    let finished_count = Arc::new(AtomicUsize::new(0));

    for (_, task) in items.into_iter() {
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
