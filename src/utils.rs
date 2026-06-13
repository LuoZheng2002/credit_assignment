use serde::{Serialize, de::DeserializeOwned};
use std::{
    fs::{File, OpenOptions},
    future::Future,
    io::BufReader,
    path::{Path, PathBuf},
    sync::{OnceLock, RwLock},
};

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

pub fn block_on_async<F: Future>(future: F) -> F::Output {
    if let Ok(handle) = tokio::runtime::Handle::try_current() {
        tokio::task::block_in_place(|| handle.block_on(future))
    } else {
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(future)
    }
}

pub fn load_jinja_template_environment(
    template_path: &str,
    template_name: &'static str,
) -> Result<minijinja::Environment<'static>, String> {
    let template_source = std::fs::read_to_string(template_path)
        .map_err(|err| format!("Failed to read {}: {}", template_path, err))?;
    let mut env = minijinja::Environment::new();
    env.add_template_owned(template_name, template_source)
        .map_err(|err| format!("Failed to parse {} template: {}", template_name, err))?;
    Ok(env)
}

pub fn extract_boxed_content(text: &str) -> Option<String> {
    const MARKER: &str = "\\boxed{";
    let mut search_start = 0usize;

    while let Some(relative_start) = text[search_start..].find(MARKER) {
        let start = search_start + relative_start;
        let after_marker = start + MARKER.len();

        let mut bracket_depth = 1;
        let mut content = String::new();
        let mut end_index_after_closing_brace: Option<usize> = None;

        for (offset, ch) in text[after_marker..].char_indices() {
            match ch {
                '{' => {
                    bracket_depth += 1;
                    content.push(ch);
                }
                '}' => {
                    bracket_depth -= 1;
                    if bracket_depth == 0 {
                        end_index_after_closing_brace = Some(after_marker + offset + ch.len_utf8());
                        break;
                    }
                    content.push(ch);
                }
                other => content.push(other),
            }
        }

        if end_index_after_closing_brace.is_none() {
            return None;
        }

        if !content.trim().is_empty() {
            return Some(content);
        }

        search_start = end_index_after_closing_brace.unwrap();
    }

    None
}

// pub fn storage_dir_from_env() -> Result<String, String> {
//     static STORAGE_DIR: LazyLock<Result<String, String>> = LazyLock::new(|| {
//         dotenvy::dotenv().ok();
//         let storage_dir = std::env::var("STORAGE_DIR")
//             .map_err(|err| format!("Failed to read STORAGE_DIR from environment: {}", err))?;
//         let storage_dir = storage_dir.trim();
//         if storage_dir.is_empty() {
//             return Err("STORAGE_DIR is set but empty".to_string());
//         }
//         Ok(storage_dir.to_string())
//     });

//     STORAGE_DIR
//         .as_ref()
//         .map(|storage_dir| storage_dir.clone())
//         .map_err(|err| err.clone())
// }

static STORAGE_MOUNT_DIR: OnceLock<RwLock<String>> = OnceLock::new();

fn mount_dir_lock() -> &'static RwLock<String> {
    STORAGE_MOUNT_DIR.get_or_init(|| RwLock::new(String::new()))
}

fn _normalize_non_empty_dir(raw: &str, arg_name: &str) -> Result<String, String> {
    let value = raw.trim();
    if value.is_empty() {
        return Err(format!("{} must be non-empty", arg_name));
    }
    Ok(value.to_string())
}

pub fn configure_mount_dir(mount_dir: &str) -> Result<(), String> {
    let mount_dir = _normalize_non_empty_dir(mount_dir, "--mount-dir")?;
    let mut guard = mount_dir_lock()
        .write()
        .map_err(|err| format!("failed to set mount dir: {}", err))?;
    *guard = mount_dir;
    Ok(())
}

pub fn mount_dir() -> Result<String, String> {
    let guard = mount_dir_lock()
        .read()
        .map_err(|err| format!("failed to read mount dir: {}", err))?;
    if guard.is_empty() {
        return Err("MOUNT_DIR not set".into());
    }
    Ok(guard.clone())
}

fn storage_dir_from_mount_dir(subdir: &str) -> Result<String, String> {
    Ok(PathBuf::from(mount_dir()?)
        .join(subdir)
        .display()
        .to_string())
}

pub fn storage_large_files_dir() -> Result<String, String> {
    storage_dir_from_mount_dir("large_files")
}

pub fn storage_medium_files_dir() -> Result<String, String> {
    storage_dir_from_mount_dir("medium_files")
}

pub fn storage_small_files_dir() -> Result<String, String> {
    storage_dir_from_mount_dir("small_files")
}
