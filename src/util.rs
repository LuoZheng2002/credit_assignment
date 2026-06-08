use std::{future::Future, sync::LazyLock};

pub fn block_on_async<F: Future>(future: F) -> F::Output {
    if let Ok(handle) = tokio::runtime::Handle::try_current() {
        tokio::task::block_in_place(|| handle.block_on(future))
    } else {
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(future)
    }
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

pub fn storage_dir_from_env() -> Result<String, String> {
    static STORAGE_DIR: LazyLock<Result<String, String>> = LazyLock::new(|| {
        dotenvy::dotenv().ok();
        let storage_dir = std::env::var("STORAGE_DIR")
            .map_err(|err| format!("Failed to read STORAGE_DIR from environment: {}", err))?;
        let storage_dir = storage_dir.trim();
        if storage_dir.is_empty() {
            return Err("STORAGE_DIR is set but empty".to_string());
        }
        Ok(storage_dir.to_string())
    });

    STORAGE_DIR
        .as_ref()
        .map(|storage_dir| storage_dir.clone())
        .map_err(|err| err.clone())
}

pub fn hpc_training_root_dir_from_env() -> Result<String, String> {
    static HPC_TRAINING_ROOT_DIR: LazyLock<Result<String, String>> = LazyLock::new(|| {
        dotenvy::dotenv().ok();
        let root_dir = std::env::var("HPC_TRAINING_ROOT_DIR").map_err(|err| {
            format!(
                "Failed to read HPC_TRAINING_ROOT_DIR from environment: {}",
                err
            )
        })?;
        let root_dir = root_dir.trim();
        if root_dir.is_empty() {
            return Err("HPC_TRAINING_ROOT_DIR is set but empty".to_string());
        }
        Ok(root_dir.to_string())
    });

    HPC_TRAINING_ROOT_DIR
        .as_ref()
        .map(|root_dir| root_dir.clone())
        .map_err(|err| err.clone())
}
