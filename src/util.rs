pub fn extract_boxed_content(text: &str) -> Option<String> {
    const MARKER: &str = "\\boxed{";
    let start = text.find(MARKER)?;
    let mut bracket_depth = 1;
    let mut content = String::new();
    for ch in text[start + MARKER.len()..].chars() {
        match ch {
            '{' => {
                bracket_depth += 1;
                content.push(ch);
            }
            '}' => {
                bracket_depth -= 1;
                if bracket_depth == 0 {
                    if content.trim().is_empty() {
                        return None;
                    } else {
                        return Some(content);
                    }
                }
                content.push(ch);
            }
            other => content.push(other),
        }
    }
    None
}
