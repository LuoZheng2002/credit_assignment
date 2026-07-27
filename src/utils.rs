use serde::{Serialize, de::DeserializeOwned};
use std::{
    env,
    fs::{File, OpenOptions},
    future::Future,
    io::BufReader,
    path::{Path, PathBuf},
    process::Command,
    sync::OnceLock,
};
use tokenizers::Tokenizer;

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
    api_name: &str,
) -> Result<minijinja::Environment<'static>, String> {
    // JIT: if the template file doesn't exist, try to download the tokenizer first
    // (the download script creates both tokenizer.json and chat_template.jinja).
    if !Path::new(template_path).exists() {
        ensure_tokenizer_files(api_name)?;
    }

    let template_source = std::fs::read_to_string(template_path)
        .map_err(|err| format!("Failed to read {}: {}", template_path, err))?;
    let mut env = minijinja::Environment::new();
    env.add_template_owned(template_name, template_source)
        .map_err(|err| format!("Failed to parse {} template: {}", template_name, err))?;
    Ok(env)
}

pub fn load_tokenizer_from_local_or_hf(local_tokenizer_path: &str, api_name: &str) -> Tokenizer {
    // JIT: if the local tokenizer directory doesn't exist, run the download script first.
    let _ = ensure_tokenizer_files(api_name);

    if Path::new(local_tokenizer_path).exists() {
        return Tokenizer::from_file(local_tokenizer_path).unwrap_or_else(|err| {
            panic!(
                "Failed to load local tokenizer from {}: {}",
                local_tokenizer_path, err
            )
        });
    }

    let token = env::var("HF_TOKEN")
        .ok()
        .or_else(|| env::var("HUGGINGFACE_HUB_TOKEN").ok());
    let params = tokenizers::FromPretrainedParameters {
        token,
        ..Default::default()
    };

    Tokenizer::from_pretrained(api_name, Some(params)).unwrap_or_else(|err| {
        panic!(
            "Failed to load tokenizer for {}. Looked for local file {} first, then fell back to Hugging Face: {}",
            api_name, local_tokenizer_path, err
        )
    })
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

static STORAGE_MOUNT_DIR: OnceLock<std::sync::RwLock<String>> = OnceLock::new();

fn mount_dir_lock() -> &'static std::sync::RwLock<String> {
    STORAGE_MOUNT_DIR.get_or_init(|| std::sync::RwLock::new(String::new()))
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

// ---------------------------------------------------------------------------
// JIT (Just-In-Time) resource helpers
// ---------------------------------------------------------------------------

/// Root directory of the repository, derived from `CARGO_MANIFEST_DIR` at compile time.
pub(crate) fn repo_root() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
}

/// Blocking helper that runs a lightweight Python script via the minimal uv project.
pub(crate) fn run_python_script(script_path: &str, args: &[&str]) -> Result<(), String> {
    let mut command = Command::new("uv");
    command
        .arg("run")
        .arg("--project")
        .arg("pyprojects/minimal")
        .arg("python")
        .arg(script_path)
        .current_dir(repo_root());

    for arg in args {
        command.arg(arg);
    }

    let output = command.output().map_err(|err| {
        format!(
            "Failed to execute 'uv run --project pyprojects/minimal python {}': {}",
            script_path, err
        )
    })?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stdout = String::from_utf8_lossy(&output.stdout);
        return Err(format!(
            "Script '{}' failed with exit code {}\nstdout: {}\nstderr: {}",
            script_path,
            output.status.code().unwrap_or(-1),
            stdout.trim(),
            stderr.trim(),
        ));
    }

    Ok(())
}

/// Map a HuggingFace model ID to its local tokenizer subdirectory name under
/// `tokenizers/`.  Variants sharing the same tokenizer (e.g. Qwen3-4B /
/// Qwen3-0.6B) return the same directory.
fn tokenizer_dir_name(api_name: &str) -> &str {
    match api_name {
        "google/gemma-3-4b-it" => "gemma3",
        "meta-llama/Llama-3.1-8B-Instruct" => "llama31",
        "mistralai/Mistral-7B-Instruct-v0.3" => "mistral7b",
        "Qwen/Qwen2.5-7B-Instruct" => "qwen25",
        "Qwen/Qwen3-4B" | "Qwen/Qwen3-0.6B" => "qwen3",
        "Qwen/Qwen3.5-4B" | "Qwen/Qwen3.5-0.8B" => "qwen35",
        other => panic!("Unknown tokenizer model: {}", other),
    }
}

/// Ensure the tokenizer files (tokenizer.json, chat_template.jinja, etc.) exist
/// locally for the given HuggingFace model.  If they are missing the unified
/// `scripts/download_tokenizer.py` script is run to download them.
pub fn ensure_tokenizer_files(api_name: &str) -> Result<(), String> {
    let dir_name = tokenizer_dir_name(api_name);
    let tokenizer_json = format!("tokenizers/{}/tokenizer.json", dir_name);
    let chat_template = format!("tokenizers/{}/chat_template.jinja", dir_name);

    if Path::new(&tokenizer_json).exists() && Path::new(&chat_template).exists() {
        return Ok(());
    }

    run_python_script("scripts/download_tokenizer.py", &["--model", api_name])
}
