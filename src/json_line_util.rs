use indexmap::IndexMap;
use serde::{Serialize, de::DeserializeOwned};
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::Path;

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

pub fn read_json<T: DeserializeOwned>(path: impl AsRef<Path>) -> Result<T, String> {
    let file = File::open(path.as_ref())
        .map_err(|e| format!("Cannot open file {}: {}", path.as_ref().display(), e))?;
    let reader = BufReader::new(file);
    serde_json::from_reader(reader).map_err(|e| format!("Failed to parse JSON: {}", e))
}

pub fn write_json<T: Serialize>(file_path: impl AsRef<Path>, data: &T) -> Result<(), String> {
    let file_path = file_path.as_ref();
    if let Some(parent) = file_path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    let file = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(file_path)
        .map_err(|e| e.to_string())?;
    serde_json::to_writer_pretty(file, data).map_err(|e| e.to_string())
}
