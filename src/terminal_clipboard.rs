use std::io::{self, Write};

use base64::Engine;

pub fn copy_to_terminal_clipboard(text: &str) -> io::Result<()> {
    let encoded = base64::engine::general_purpose::STANDARD.encode(text.as_bytes());
    let mut stdout = io::stdout();
    write!(stdout, "\x1b]52;c;{}\x07", encoded)?;
    stdout.flush()
}
