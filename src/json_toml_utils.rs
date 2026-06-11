use serde::{Serialize, de::DeserializeOwned};
use std::fs::{File, OpenOptions};
use std::io::{BufReader};
use std::path::Path;

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

pub fn read_toml<T: DeserializeOwned>(path: impl AsRef<Path>) -> Result<T, String> {
    let content = std::fs::read_to_string(path.as_ref())
        .map_err(|e| format!("Cannot read file {}: {}", path.as_ref().display(), e))?;
    toml::from_str(&content).map_err(|e| format!("Failed to parse TOML: {}", e))
}

pub fn write_toml<T: Serialize>(file_path: impl AsRef<Path>, data: &T) -> Result<(), String> {
    let file_path = file_path.as_ref();
    if let Some(parent) = file_path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    let content =
        toml::to_string_pretty(data).map_err(|e| format!("Failed to serialize TOML: {}", e))?;
    std::fs::write(file_path, content)
        .map_err(|e| format!("Failed to write file {}: {}", file_path.display(), e))
}
