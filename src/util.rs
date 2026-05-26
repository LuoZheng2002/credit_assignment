use std::future::Future;

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
