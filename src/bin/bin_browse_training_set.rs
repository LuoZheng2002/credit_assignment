use std::{
    backtrace::Backtrace,
    collections::BTreeMap,
    error::Error,
    fs::File,
    io::{self, Cursor, Read, Stdout},
    path::{Path, PathBuf},
    process::Stdio,
    time::{Duration, Instant},
};

use clap::Parser;

use credit_assignment::{
    directories::{
        training_trajectories_oneshot_path, training_trajectories_path,
        training_trajectories_stats_oneshot_path, training_trajectories_stats_path,
    },
    hybrid_dataset::Training,
    json_toml_utils::read_json,
    llm_model::{
        Gemma3_4BIt, Llama31_8BInstruct, LlmModelMarker, LlmModelName, Mistral7BInstructV03,
        MyTokenizer, Qwen3_4B, Qwen3_06B, Qwen25_7B, Qwen35_4B, Qwen35_08B,
    },
    terminal_clipboard::copy_to_terminal_clipboard,
    training_set::{
        DirectTrainingSetStatistics, DirectTrainingTrajectory, TrainingTrajectoryConfigBundle,
        open_training_trajectories,
    },
    utils::configure_mount_dir,
};
use crossterm::cursor::Show;
use crossterm::event::{
    self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode, KeyEvent, MouseEvent,
    MouseEventKind,
};
use crossterm::execute;
use crossterm::terminal::{
    EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
};
use ratatui::Terminal;
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Direction, Layout, Position, Rect};
use ratatui::prelude::Widget;
use ratatui::style::{Color, Style};
use ratatui::text::{Line, Span, Text};
use ratatui::widgets::{Block, Borders, Paragraph, Wrap};
use ratatui_core::buffer::Buffer;
use serde::Deserialize;
use tokio::process::Command;

#[derive(Parser, Debug)]
#[command(author, version, about = "Interactively browse training sets")]
struct Args {
    #[arg(value_enum, short, long)]
    model_cli_name: LlmModelName,
    #[arg(long)]
    config_nickname: String,
    #[arg(long)]
    epoch: usize, // the epoch index
    #[arg(long, default_value = "results")]
    mount_dir: String,
    #[arg(long)]
    summary_only: bool,
    #[arg(long, default_value_t = 8)]
    sample_count: usize,
    #[arg(long)]
    oneshot: bool,
    #[arg(long)]
    dump_decoded_samples: bool,
    #[arg(long, value_delimiter = ',')]
    sample_indices: Vec<usize>,
    #[arg(long, default_value_t = 6000)]
    decoded_char_limit: usize,
}

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn training_set_file_path(
    mount_dir: &str,
    model_cli_name: &str,
    config_nickname: &str,
    epoch: usize,
    oneshot: bool,
) -> PathBuf {
    let path = if oneshot {
        training_trajectories_oneshot_path(mount_dir, model_cli_name, config_nickname)
    } else {
        training_trajectories_path(mount_dir, model_cli_name, config_nickname, epoch)
    };
    repo_root().join(path)
}

fn training_set_config_bundle_file_path(training_set_path: &Path) -> Result<PathBuf, String> {
    if training_set_path.is_dir() || training_set_path.extension().is_none() {
        Ok(training_set_path.join("config_bundle.json"))
    } else {
        training_set_path
            .parent()
            .map(|parent| parent.join("config_bundle.json"))
            .ok_or_else(|| {
                format!(
                    "Cannot derive config_bundle.json path from {}",
                    training_set_path.display()
                )
            })
    }
}

async fn download_training_set(
    model_cli_name: &str,
    config_nickname: &str,
    epoch: usize,
) -> Result<(), Box<dyn Error>> {
    let mut command = Command::new("uv");
    command
        .arg("run")
        .arg("--project")
        .arg("pyprojects/minimal")
        .arg("python")
        .arg("scripts/download_training_set.py")
        .arg("--model-cli-name")
        .arg(model_cli_name)
        .arg("--config-nickname")
        .arg(config_nickname)
        .arg("--epoch")
        .arg(epoch.to_string())
        .current_dir(repo_root())
        .stdin(Stdio::inherit())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit());

    let status = command.status().await?;
    if !status.success() {
        return Err(format!(
            "download_training_set.py failed with status {} while fetching training set for {}/{}/epoch_{}",
            status, model_cli_name, config_nickname, epoch
        )
        .into());
    }

    Ok(())
}

struct ConversationRender {
    plain: String,
    styled: Text<'static>,
}

struct LoadedTrajectory<M: LlmModelMarker> {
    index: usize,
    key: usize,
    trajectory: DirectTrainingTrajectory<M>,
    conversation_render: ConversationRender,
}

const CONVERSATION_SCROLL_SENSITIVITY: usize = 4;
const MOUSE_SCROLL_DEBOUNCE: Duration = Duration::from_millis(100);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PaneFocus {
    Summary,
    Conversation,
}

impl PaneFocus {
    fn next(self) -> Self {
        match self {
            Self::Summary => Self::Conversation,
            Self::Conversation => Self::Summary,
        }
    }
}

struct App<M: LlmModelMarker> {
    store: Vec<DirectTrainingTrajectory<M>>,
    keys: Vec<usize>,
    selected_index: usize,
    focus: PaneFocus,
    summary_scroll: usize,
    summary_max_scroll: usize,
    conversation_scroll: usize,
    conversation_max_scroll: usize,
    summary_area: Option<Rect>,
    conversation_area: Option<Rect>,
    jump_input_area: Option<Rect>,
    controls_area: Option<Rect>,
    loaded: Option<LoadedTrajectory<M>>,
    statistics: Option<DirectTrainingSetStatistics>,
    jump_input: String,
    jump_input_active: bool,
    jump_status: Option<String>,
    last_mouse_scroll_at: Option<Instant>,
}

impl<M: LlmModelMarker> App<M> {
    fn new(
        store: Vec<DirectTrainingTrajectory<M>>,
        keys: Vec<usize>,
        statistics: Option<DirectTrainingSetStatistics>,
    ) -> Self {
        Self {
            store,
            keys,
            selected_index: 0,
            focus: PaneFocus::Conversation,
            summary_scroll: 0,
            summary_max_scroll: 0,
            conversation_scroll: 0,
            conversation_max_scroll: 0,
            summary_area: None,
            conversation_area: None,
            jump_input_area: None,
            controls_area: None,
            loaded: None,
            statistics,
            jump_input: String::new(),
            jump_input_active: false,
            jump_status: None,
            last_mouse_scroll_at: None,
        }
    }

    fn ensure_selected_loaded(&mut self) {
        if self.keys.is_empty() {
            self.loaded = None;
            return;
        }
        if self
            .loaded
            .as_ref()
            .is_some_and(|loaded| loaded.index == self.selected_index)
        {
            return;
        }

        let key = self.keys[self.selected_index];
        let trajectory = self
            .store
            .get(key)
            .cloned()
            .expect("key from training trajectory array must exist");

        let conversation_render = build_conversation_render::<M>(&trajectory);
        self.loaded = Some(LoadedTrajectory {
            index: self.selected_index,
            key,
            trajectory,
            conversation_render,
        });
        self.summary_scroll = 0;
        self.conversation_scroll = 0;
    }

    fn draw(&mut self, frame: &mut ratatui::Frame<'_>) {
        self.ensure_selected_loaded();

        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(12),
                Constraint::Min(5),
                Constraint::Length(3),
                Constraint::Length(3),
            ])
            .split(frame.area());

        let summary_text = if self.keys.is_empty() {
            "No training trajectories found".to_string()
        } else {
            let loaded = self
                .loaded
                .as_ref()
                .expect("selected trajectory must be loaded");
            let trainable_tokens = loaded
                .trajectory
                .labels
                .iter()
                .filter(|&&label| label != -100)
                .count();
            let mut text = format!(
                "Model: {}\nTrajectory: {}/{} (key #{})\nQuestion #{} [{}:{}]\nAverage absolute segment advantage: {:.6}\nTrajectory length (tokens): {}\nTrainable tokens: {}",
                M::CLI_NAME,
                loaded.index + 1,
                self.keys.len(),
                loaded.key,
                loaded.trajectory.question.flat_id.0,
                loaded.trajectory.question.dataset_name,
                loaded.trajectory.question.question_id,
                loaded.trajectory.average_absolute_segment_advantage,
                loaded.trajectory.input_ids.len(),
                trainable_tokens,
            );
            let trajectory_advantage_values =
                compress_advantage_changes(&loaded.trajectory.advantages);
            text.push_str(&format!(
                "\nAdvantage values (change points): {:?}",
                trajectory_advantage_values
            ));
            if let Some(statistics) = &self.statistics {
                text.push_str(&format!(
                    "\n\n[training set stats] total: {}  adopted: {}\nmax avg absolute advantage: {:.6}  cutoff: {:.6}  min avg absolute advantage: {:.6}\nadv balance pre(+/-): {:.6} / {:.6}  mult(+/-): {:.6} / {:.6}  post(+/-): {:.6} / {:.6}",
                    statistics.total_trajectories,
                    statistics.adopted_trajectories,
                    statistics.max_average_absolute_advantage,
                    statistics.average_absolute_advantage_cutoff,
                    statistics.min_average_absolute_advantage,
                    statistics.pre_balance_total_positive_advantage,
                    statistics.pre_balance_total_negative_advantage_magnitude,
                    statistics.positive_advantage_multiplier,
                    statistics.negative_advantage_multiplier,
                    statistics.post_balance_total_positive_advantage,
                    statistics.post_balance_total_negative_advantage_magnitude,
                ));
            }
            text
        };

        let summary_block = Block::default()
            .borders(Borders::ALL)
            .title("Metadata")
            .border_style(if self.focus == PaneFocus::Summary {
                Style::default().fg(Color::LightGreen)
            } else {
                Style::default()
            });
        let summary_inner = summary_block.inner(chunks[0]);
        let summary_line_count = count_wrapped_lines(&summary_text, summary_inner);
        let summary_height = summary_inner.height.max(1) as usize;
        self.summary_max_scroll = bottom_scroll_limit(summary_line_count, summary_height);
        if self.summary_scroll > self.summary_max_scroll {
            self.summary_scroll = self.summary_max_scroll;
        }

        frame.render_widget(
            Paragraph::new(summary_text)
                .wrap(Wrap { trim: false })
                .scroll((clamp_scroll(self.summary_scroll), 0))
                .block(summary_block),
            chunks[0],
        );
        self.summary_area = Some(chunks[0]);

        if self.keys.is_empty() {
            frame.render_widget(
                Paragraph::new("No conversation to render")
                    .block(Block::default().borders(Borders::ALL).title("Conversation")),
                chunks[1],
            );
            self.conversation_area = Some(chunks[1]);
        } else {
            let loaded = self
                .loaded
                .as_ref()
                .expect("selected trajectory must be loaded");
            self.conversation_area = Some(chunks[1]);
            let conversation_block = Block::default()
                .borders(Borders::ALL)
                .title("Conversation")
                .border_style(if self.focus == PaneFocus::Conversation {
                    Style::default().fg(Color::LightGreen)
                } else {
                    Style::default()
                });
            let conversation_inner = conversation_block.inner(chunks[1]);
            let conversation_line_count =
                count_wrapped_lines(&loaded.conversation_render.plain, conversation_inner);
            let conversation_height = conversation_inner.height.max(1) as usize;
            self.conversation_max_scroll =
                bottom_scroll_limit(conversation_line_count, conversation_height);
            if self.conversation_scroll > self.conversation_max_scroll {
                self.conversation_scroll = self.conversation_max_scroll;
            }
            frame.render_widget(
                Paragraph::new(loaded.conversation_render.styled.clone())
                    .wrap(Wrap { trim: false })
                    .scroll((clamp_scroll(self.conversation_scroll), 0))
                    .block(conversation_block),
                chunks[1],
            );
        }

        let jump_hint = if self.jump_input_active {
            "Enter key and press Enter (Esc to cancel)"
        } else {
            "Press '/' to jump by key"
        };
        let jump_status = self.jump_status.clone().unwrap_or_default();
        let jump_text = if jump_status.is_empty() {
            format!("{}{}", jump_hint, self.jump_input)
        } else {
            format!("{}{}  [{}]", jump_hint, self.jump_input, jump_status)
        };
        frame.render_widget(
            Paragraph::new(jump_text).block(
                Block::default()
                    .borders(Borders::ALL)
                    .title("Jump To Key")
                    .border_style(if self.jump_input_active {
                        Style::default().fg(Color::LightGreen)
                    } else {
                        Style::default()
                    }),
            ),
            chunks[2],
        );
        self.jump_input_area = Some(chunks[2]);

        let controls = Paragraph::new(
            "[Copy] c: copy LaTeX trajectory  Left/Right: prev/next  /: jump  Tab: focus  Up/Down/PgUp/PgDn/Home/End: scroll  q: quit",
        )
        .block(Block::default().borders(Borders::ALL).title("Controls"));
        frame.render_widget(controls, chunks[3]);
        self.controls_area = Some(chunks[3]);
    }

    fn handle_key(&mut self, key: KeyEvent) -> bool {
        if self.jump_input_active {
            return self.handle_jump_input_key(key);
        }

        match key.code {
            KeyCode::Char('q') | KeyCode::Esc => true,
            KeyCode::Char('/') => {
                self.jump_input_active = true;
                self.jump_status = None;
                false
            }
            KeyCode::Char('c') => {
                self.copy_current_trajectory_latex();
                false
            }
            KeyCode::Char(ch) if ch.is_ascii_digit() => {
                self.jump_input_active = true;
                self.jump_input.push(ch);
                self.jump_status = None;
                false
            }
            KeyCode::Tab => {
                self.focus = self.focus.next();
                false
            }
            KeyCode::Left => {
                if !self.keys.is_empty() {
                    self.selected_index = self.selected_index.saturating_sub(1);
                }
                false
            }
            KeyCode::Right => {
                if !self.keys.is_empty() {
                    self.selected_index = (self.selected_index + 1).min(self.keys.len() - 1);
                }
                false
            }
            KeyCode::Up => {
                self.scroll_focused_up(1);
                false
            }
            KeyCode::Down => {
                self.scroll_focused_down(1);
                false
            }
            KeyCode::PageUp => {
                self.scroll_focused_up(10);
                false
            }
            KeyCode::PageDown => {
                self.scroll_focused_down(10);
                false
            }
            KeyCode::Home => {
                match self.focus {
                    PaneFocus::Summary => self.summary_scroll = 0,
                    PaneFocus::Conversation => self.conversation_scroll = 0,
                }
                false
            }
            KeyCode::End => {
                match self.focus {
                    PaneFocus::Summary => self.summary_scroll = self.summary_max_scroll,
                    PaneFocus::Conversation => {
                        self.conversation_scroll = self.conversation_max_scroll
                    }
                }
                false
            }
            _ => false,
        }
    }

    fn handle_jump_input_key(&mut self, key: KeyEvent) -> bool {
        match key.code {
            KeyCode::Esc => {
                self.jump_input_active = false;
                self.jump_input.clear();
                self.jump_status = None;
                false
            }
            KeyCode::Enter => {
                if self.keys.is_empty() {
                    self.jump_status = Some("No keys in training set".to_string());
                    self.jump_input_active = false;
                    self.jump_input.clear();
                    return false;
                }

                if self.jump_input.is_empty() {
                    self.jump_status = Some("Input is empty".to_string());
                    self.jump_input_active = false;
                    return false;
                }

                match self.jump_input.parse::<usize>() {
                    Ok(target_key) => match self.keys.binary_search(&target_key) {
                        Ok(index) => {
                            self.selected_index = index;
                            self.jump_status = Some(format!("Jumped to key {}", target_key));
                        }
                        Err(_) => {
                            self.jump_status = Some(format!("Key {} not found", target_key));
                        }
                    },
                    Err(_) => {
                        self.jump_status = Some("Invalid key".to_string());
                    }
                }

                self.jump_input_active = false;
                self.jump_input.clear();
                false
            }
            KeyCode::Backspace => {
                self.jump_input.pop();
                false
            }
            KeyCode::Char(ch) if ch.is_ascii_digit() => {
                self.jump_input.push(ch);
                false
            }
            _ => false,
        }
    }

    fn handle_mouse(&mut self, mouse: MouseEvent) {
        if let Some(summary_area) = self.summary_area {
            if contains_point(summary_area, mouse.column, mouse.row) {
                self.focus = PaneFocus::Summary;
            }
        }
        if let Some(conversation_area) = self.conversation_area {
            if contains_point(conversation_area, mouse.column, mouse.row) {
                self.focus = PaneFocus::Conversation;
            }
        }
        if let Some(jump_input_area) = self.jump_input_area {
            if contains_point(jump_input_area, mouse.column, mouse.row) {
                self.jump_input_active = true;
                self.jump_status = None;
            }
        }
        if let Some(controls_area) = self.controls_area {
            if contains_point(controls_area, mouse.column, mouse.row)
                && matches!(mouse.kind, MouseEventKind::Down(_))
            {
                self.copy_current_trajectory_latex();
                return;
            }
        }

        match mouse.kind {
            MouseEventKind::ScrollUp => {
                if !self.should_process_mouse_scroll() {
                    return;
                }
                self.scroll_focused_up(1)
            }
            MouseEventKind::ScrollDown => {
                if !self.should_process_mouse_scroll() {
                    return;
                }
                self.scroll_focused_down(1)
            }
            _ => {}
        }
    }

    fn should_process_mouse_scroll(&mut self) -> bool {
        let now = Instant::now();
        if self
            .last_mouse_scroll_at
            .is_some_and(|last| now.saturating_duration_since(last) < MOUSE_SCROLL_DEBOUNCE)
        {
            return false;
        }
        self.last_mouse_scroll_at = Some(now);
        true
    }

    fn scroll_focused_up(&mut self, magnitude: usize) {
        match self.focus {
            PaneFocus::Summary => {
                self.summary_scroll = self.summary_scroll.saturating_sub(magnitude)
            }
            PaneFocus::Conversation => {
                let scaled = magnitude.saturating_mul(CONVERSATION_SCROLL_SENSITIVITY);
                self.conversation_scroll = self.conversation_scroll.saturating_sub(scaled)
            }
        }
    }

    fn scroll_focused_down(&mut self, magnitude: usize) {
        match self.focus {
            PaneFocus::Summary => {
                self.summary_scroll = self.summary_scroll.saturating_add(magnitude)
            }
            PaneFocus::Conversation => {
                let scaled = magnitude.saturating_mul(CONVERSATION_SCROLL_SENSITIVITY);
                self.conversation_scroll = self.conversation_scroll.saturating_add(scaled)
            }
        }
    }

    fn copy_current_trajectory_latex(&mut self) {
        self.ensure_selected_loaded();
        let Some(loaded) = &self.loaded else {
            self.jump_status = Some("Nothing to copy".to_string());
            return;
        };
        let latex = build_latex_conversation_export::<M>(&loaded.trajectory);
        match copy_to_terminal_clipboard(&latex) {
            Ok(()) => self.jump_status = Some("Copied LaTeX trajectory".to_string()),
            Err(error) => self.jump_status = Some(format!("Copy failed: {error}")),
        }
    }
}

fn clamp_scroll(value: usize) -> u16 {
    if value > u16::MAX as usize {
        u16::MAX
    } else {
        value as u16
    }
}

fn contains_point(area: Rect, x: u16, y: u16) -> bool {
    x >= area.x && x < area.x + area.width && y >= area.y && y < area.y + area.height
}

fn count_wrapped_lines(text: &str, area: Rect) -> usize {
    if area.width == 0 {
        return 0;
    }
    let height = area.height.max(1).saturating_add(1024).min(u16::MAX);
    let mut buffer = Buffer::empty(Rect::new(0, 0, area.width, height));
    Paragraph::new(text)
        .wrap(Wrap { trim: false })
        .render(buffer.area, &mut buffer);
    let mut last_non_empty = None;
    for y in 0..height {
        let row_has_content = (0..area.width).any(|x| {
            buffer
                .cell(Position::new(x, y))
                .is_some_and(|cell| !cell.symbol().trim().is_empty())
        });
        if row_has_content {
            last_non_empty = Some(y);
        }
    }
    last_non_empty
        .map(|last| (last + 1) as usize)
        .unwrap_or(0)
        .max(1)
}

fn bottom_scroll_limit(lines: usize, height: usize) -> usize {
    if height == 0 {
        return lines + 2;
    }
    let extra = lines.saturating_add(2);
    extra.saturating_sub(height)
}

fn advantage_to_color(advantage: f32) -> Color {
    let x = (advantage.clamp(-3.0, 3.0)) / 3.0;
    let (red, green) = if x >= 0.0 {
        (((1.0 - x) * 255.0).round() as u8, 255)
    } else {
        (255, ((1.0 + x) * 255.0).round() as u8)
    };
    Color::Rgb(red, green, 0)
}

fn color_to_rgb(color: Color) -> (u8, u8, u8) {
    match color {
        Color::Rgb(red, green, blue) => (red, green, blue),
        Color::Black => (0, 0, 0),
        Color::Red => (255, 0, 0),
        Color::Green => (0, 255, 0),
        Color::Yellow => (255, 255, 0),
        Color::Blue => (0, 0, 255),
        Color::Magenta => (255, 0, 255),
        Color::Cyan => (0, 255, 255),
        Color::Gray | Color::DarkGray => (128, 128, 128),
        Color::LightRed => (255, 96, 96),
        Color::LightGreen => (96, 255, 96),
        Color::LightYellow => (255, 255, 96),
        Color::LightBlue => (96, 96, 255),
        Color::LightMagenta => (255, 96, 255),
        Color::LightCyan => (96, 255, 255),
        Color::White => (255, 255, 255),
        _ => (255, 255, 255),
    }
}

fn latex_escape_char(ch: char, output: &mut String) {
    match ch {
        '\\' => output.push_str("\\textbackslash{}"),
        '{' => output.push_str("\\{"),
        '}' => output.push_str("\\}"),
        '$' => output.push_str("\\$"),
        '&' => output.push_str("\\&"),
        '#' => output.push_str("\\#"),
        '_' => output.push_str("\\_"),
        '%' => output.push_str("\\%"),
        '^' => output.push_str("\\textasciicircum{}"),
        '~' => output.push_str("\\textasciitilde{}"),
        '\n' => output.push_str("\\\\\n"),
        ' ' => output.push('~'),
        '\t' => output.push_str("\\quad{}"),
        _ => output.push(ch),
    }
}

fn latex_escape_text(text: &str) -> String {
    let mut output = String::new();
    for ch in text.chars() {
        latex_escape_char(ch, &mut output);
    }
    output
}

fn push_latex_colored_group(output: &mut String, rgb: Option<(u8, u8, u8)>, text: &str) {
    if text.is_empty() {
        return;
    }
    let escaped = latex_escape_text(text);
    if let Some((red, green, blue)) = rgb {
        output.push_str(&format!(
            "\\textcolor[RGB]{{{red},{green},{blue}}}{{{escaped}}}"
        ));
    } else {
        output.push_str(&escaped);
    }
}

fn build_latex_conversation_export<M: LlmModelMarker>(
    trajectory: &DirectTrainingTrajectory<M>,
) -> String {
    let mut output = String::from("\\begingroup\\ttfamily\\small\n");
    let mut current_rgb: Option<(u8, u8, u8)> = None;
    let mut current_text = String::new();

    for index in 0..trajectory.input_ids.len() {
        let token_id = trajectory.input_ids[index];
        let label = trajectory.labels[index];
        let advantage = trajectory.advantages[index];
        let token_text = <M::Tokenizer as MyTokenizer<M>>::decode_i32_ids(&[token_id]);
        let token_rgb = if label == -100 {
            None
        } else {
            Some(color_to_rgb(advantage_to_color(advantage)))
        };
        if token_rgb != current_rgb {
            push_latex_colored_group(&mut output, current_rgb, &current_text);
            current_text.clear();
            current_rgb = token_rgb;
        }
        current_text.push_str(&token_text);
    }
    push_latex_colored_group(&mut output, current_rgb, &current_text);
    output.push_str("\n\\endgroup\n");
    output
}

fn compress_advantage_changes(advantages: &[f32]) -> Vec<f32> {
    let mut compressed = Vec::new();
    for &value in advantages {
        if compressed.last().is_none_or(|last| *last != value) {
            compressed.push(value);
        }
    }
    compressed
}

fn push_text_as_spans(
    lines: &mut Vec<Vec<Span<'static>>>,
    text: &str,
    style: Style,
    plain: &mut String,
) {
    if text.is_empty() {
        return;
    }
    if lines.is_empty() {
        lines.push(Vec::new());
    }

    for part in text.split_inclusive('\n') {
        let has_newline = part.ends_with('\n');
        let content = if has_newline {
            &part[..part.len() - 1]
        } else {
            part
        };
        if !content.is_empty() {
            lines
                .last_mut()
                .expect("lines must not be empty")
                .push(Span::styled(content.to_string(), style));
        }
        plain.push_str(content);
        if has_newline {
            plain.push('\n');
            lines.push(Vec::new());
        }
    }
}

fn build_conversation_render<M: LlmModelMarker>(
    trajectory: &DirectTrainingTrajectory<M>,
) -> ConversationRender {
    let mut lines: Vec<Vec<Span<'static>>> = vec![Vec::new()];
    let mut plain = String::new();

    assert_eq!(
        trajectory.input_ids.len(),
        trajectory.labels.len(),
        "input_ids and labels must have same length"
    );
    assert_eq!(
        trajectory.input_ids.len(),
        trajectory.advantages.len(),
        "input_ids and advantages must have same length"
    );

    for index in 0..trajectory.input_ids.len() {
        let token_id = trajectory.input_ids[index];
        let label = trajectory.labels[index];
        let advantage = trajectory.advantages[index];
        let token_text = <M::Tokenizer as MyTokenizer<M>>::decode_i32_ids(&[token_id]);
        let style = if label == -100 {
            Style::default().fg(Color::White)
        } else {
            Style::default().fg(advantage_to_color(advantage))
        };
        push_text_as_spans(&mut lines, &token_text, style, &mut plain);
    }

    if lines.is_empty() {
        lines.push(Vec::new());
    }

    let styled_lines: Vec<Line<'static>> = lines.into_iter().map(Line::from).collect();
    ConversationRender {
        plain,
        styled: Text::from(styled_lines),
    }
}

fn run_app<M: LlmModelMarker>(
    terminal: &mut Terminal<CrosstermBackend<Stdout>>,
    mut app: App<M>,
) -> Result<(), Box<dyn Error>> {
    loop {
        terminal.draw(|frame| app.draw(frame))?;
        match event::read()? {
            Event::Key(key) => {
                if app.handle_key(key) {
                    break;
                }
            }
            Event::Mouse(mouse) => app.handle_mouse(mouse),
            _ => {}
        }
    }
    Ok(())
}

struct RunProgramArgs {
    config_nickname: String,
    epoch: usize, // the epoch index
    mount_dir: String,
    summary_only: bool,
    sample_count: usize,
    oneshot: bool,
    dump_decoded_samples: bool,
    sample_indices: Vec<usize>,
    decoded_char_limit: usize,
}

fn restore_terminal_after_panic() {
    let _ = disable_raw_mode();
    let mut stderr = io::stderr();
    let _ = execute!(stderr, LeaveAlternateScreen, DisableMouseCapture, Show);
}

#[tokio::main]
async fn main() {
    std::panic::set_hook(Box::new(|info| {
        restore_terminal_after_panic();
        eprintln!("panic occurred: {}", info);
        let rust_backtrace = std::env::var("RUST_BACKTRACE").ok();
        if matches!(rust_backtrace.as_deref(), Some("1") | Some("full")) {
            let backtrace = Backtrace::force_capture();
            eprintln!("backtrace:\n{}", backtrace);
        }
        std::process::abort();
    }));
    dotenvy::dotenv().ok();
    let Args {
        model_cli_name: model,
        config_nickname,
        epoch,
        mount_dir,
        summary_only,
        sample_count,
        oneshot,
        dump_decoded_samples,
        sample_indices,
        decoded_char_limit,
    } = Args::parse();
    configure_mount_dir(&mount_dir)
        .unwrap_or_else(|err| panic!("failed to configure mount dir for browsing: {}", err));

    let model_cli_name = model.cli_name();
    let training_set_path = training_set_file_path(
        &mount_dir,
        &model_cli_name,
        &config_nickname,
        epoch,
        oneshot,
    );
    let training_set_msgpack_path = training_set_path.join("trajectories.msgpack");
    let training_set_stats_path = repo_root().join(if oneshot {
        training_trajectories_stats_oneshot_path(&mount_dir, &model_cli_name, &config_nickname)
    } else {
        training_trajectories_stats_path(&mount_dir, &model_cli_name, &config_nickname, epoch)
    });
    let training_set_config_bundle_path =
        training_set_config_bundle_file_path(&training_set_path).unwrap();
    let missing_required_artifacts = if summary_only {
        !training_set_msgpack_path.exists()
    } else {
        !training_set_msgpack_path.exists()
            || !training_set_stats_path.exists()
            || !training_set_config_bundle_path.exists()
    };
    if missing_required_artifacts {
        eprintln!(
            "Missing training set artifacts for {}/{}/epoch_{}; downloading...",
            model_cli_name, config_nickname, epoch
        );
        download_training_set(&model_cli_name, &config_nickname, epoch)
            .await
            .unwrap();
    }

    if !summary_only {
        let _training_trajectory_config_bundle =
            read_json::<TrainingTrajectoryConfigBundle<Training>>(&training_set_config_bundle_path)
                .unwrap();
    }
    let run_program_args = RunProgramArgs {
        config_nickname,
        epoch,
        mount_dir,
        summary_only,
        sample_count,
        oneshot,
        dump_decoded_samples,
        sample_indices,
        decoded_char_limit,
    };
    match model {
        LlmModelName::Gemma3_4b => run_program::<Gemma3_4BIt>(run_program_args).await,
        LlmModelName::Llama31_8b => run_program::<Llama31_8BInstruct>(run_program_args).await,
        LlmModelName::Mistral7bInstructV03 => {
            run_program::<Mistral7BInstructV03>(run_program_args).await
        }
        LlmModelName::Qwen3_06b => run_program::<Qwen3_06B>(run_program_args).await,
        LlmModelName::Qwen3_4b => run_program::<Qwen3_4B>(run_program_args).await,
        LlmModelName::Qwen25_7b => run_program::<Qwen25_7B>(run_program_args).await,
        LlmModelName::Qwen35_08b => run_program::<Qwen35_08B>(run_program_args).await,
        LlmModelName::Qwen35_4b => run_program::<Qwen35_4B>(run_program_args).await,
    }
}

async fn run_program<M: LlmModelMarker>(run_program_args: RunProgramArgs) {
    let RunProgramArgs {
        config_nickname,
        epoch,
        mount_dir,
        summary_only,
        sample_count,
        oneshot,
        dump_decoded_samples,
        sample_indices,
        decoded_char_limit,
    } = run_program_args;
    let training_set_store = if oneshot {
        let training_set_path =
            training_trajectories_oneshot_path(&mount_dir, M::CLI_NAME, &config_nickname);
        open_training_trajectories_file::<M>(&format!("{training_set_path}/trajectories.msgpack"))
    } else {
        open_training_trajectories::<M>(&mount_dir, &config_nickname, epoch)
    };
    let keys = (0..training_set_store.len()).collect::<Vec<usize>>();
    let statistics = read_json::<DirectTrainingSetStatistics>(repo_root().join(if oneshot {
        training_trajectories_stats_oneshot_path(&mount_dir, M::CLI_NAME, &config_nickname)
    } else {
        training_trajectories_stats_path(&mount_dir, M::CLI_NAME, &config_nickname, epoch)
    }))
    .ok();

    if summary_only {
        print_summary::<M>(
            &config_nickname,
            epoch,
            &training_set_store,
            statistics.as_ref(),
            sample_count,
            dump_decoded_samples,
            &sample_indices,
            decoded_char_limit,
        );
        return;
    }

    enable_raw_mode().unwrap();
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture).unwrap();
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend).unwrap();
    let app = App::<M>::new(training_set_store, keys, statistics);
    let result = run_app(&mut terminal, app);
    disable_raw_mode().unwrap();
    execute!(
        terminal.backend_mut(),
        LeaveAlternateScreen,
        DisableMouseCapture
    )
    .unwrap();
    terminal.show_cursor().unwrap();
    result.unwrap();
}

fn open_training_trajectories_file<M: LlmModelMarker>(
    file_path: &str,
) -> Vec<DirectTrainingTrajectory<M>> {
    let mut bytes = Vec::new();
    File::open(file_path)
        .unwrap_or_else(|err| panic!("Failed to open training trajectories at {file_path}: {err}"))
        .read_to_end(&mut bytes)
        .unwrap_or_else(|err| panic!("Failed to read training trajectories at {file_path}: {err}"));
    let mut cursor = Cursor::new(bytes.as_slice());
    let total_len = bytes.len() as u64;
    let mut trajectories = Vec::new();
    while cursor.position() < total_len {
        let mut deserializer = rmp_serde::Deserializer::new(&mut cursor);
        let trajectory = DirectTrainingTrajectory::<M>::deserialize(&mut deserializer)
            .unwrap_or_else(|err| {
                panic!("Failed to deserialize training trajectory at {file_path}: {err}")
            });
        trajectories.push(trajectory);
    }
    trajectories
}

fn percentile(sorted_values: &[usize], numerator: usize, denominator: usize) -> usize {
    if sorted_values.is_empty() {
        return 0;
    }
    let last = sorted_values.len() - 1;
    let index = (last * numerator + denominator / 2) / denominator;
    sorted_values[index.min(last)]
}

fn summarize_advantages(advantages: &[f32]) -> (usize, usize, usize, f32, f32) {
    let mut positive = 0usize;
    let mut negative = 0usize;
    let mut zero = 0usize;
    let mut min_value = f32::INFINITY;
    let mut max_value = f32::NEG_INFINITY;
    for &advantage in advantages {
        if advantage > 0.0 {
            positive += 1;
        } else if advantage < 0.0 {
            negative += 1;
        } else {
            zero += 1;
        }
        min_value = min_value.min(advantage);
        max_value = max_value.max(advantage);
    }
    if advantages.is_empty() {
        min_value = 0.0;
        max_value = 0.0;
    }
    (positive, negative, zero, min_value, max_value)
}

fn print_sample<M: LlmModelMarker>(
    label: &str,
    index: usize,
    trajectory: &DirectTrainingTrajectory<M>,
) {
    let trainable_tokens = trajectory
        .labels
        .iter()
        .filter(|&&item| item != -100)
        .count();
    let (positive, negative, zero, min_advantage, max_advantage) =
        summarize_advantages(&trajectory.advantages);
    println!(
        "{} index={} question={} dataset={} question_id={} len={} trainable={} avg_abs_adv={:.6} adv_pos={} adv_neg={} adv_zero={} adv_min={:.6} adv_max={:.6}",
        label,
        index,
        trajectory.question.flat_id.0,
        trajectory.question.dataset_name,
        trajectory.question.question_id,
        trajectory.input_ids.len(),
        trainable_tokens,
        trajectory.average_absolute_segment_advantage,
        positive,
        negative,
        zero,
        min_advantage,
        max_advantage,
    );
}

fn print_summary<M: LlmModelMarker>(
    config_nickname: &str,
    epoch: usize,
    trajectories: &[DirectTrainingTrajectory<M>],
    statistics: Option<&DirectTrainingSetStatistics>,
    sample_count: usize,
    dump_decoded_samples: bool,
    sample_indices: &[usize],
    decoded_char_limit: usize,
) {
    println!(
        "training_set model={} config={} epoch={} trajectories={}",
        M::CLI_NAME,
        config_nickname,
        epoch,
        trajectories.len()
    );
    if trajectories.is_empty() {
        return;
    }

    let mut lengths: Vec<usize> = trajectories
        .iter()
        .map(|trajectory| trajectory.input_ids.len())
        .collect();
    lengths.sort_unstable();
    let total_tokens: usize = lengths.iter().sum();
    println!(
        "lengths min={} p10={} p25={} p50={} p75={} p90={} p95={} p99={} max={} avg={:.2}",
        lengths[0],
        percentile(&lengths, 10, 100),
        percentile(&lengths, 25, 100),
        percentile(&lengths, 50, 100),
        percentile(&lengths, 75, 100),
        percentile(&lengths, 90, 100),
        percentile(&lengths, 95, 100),
        percentile(&lengths, 99, 100),
        lengths[lengths.len() - 1],
        total_tokens as f64 / lengths.len() as f64,
    );

    let mut question_counts: BTreeMap<usize, usize> = BTreeMap::new();
    let mut trainable_total = 0usize;
    let mut invalid_shape_count = 0usize;
    let mut no_trainable_count = 0usize;
    let mut total_positive_advantages = 0usize;
    let mut total_negative_advantages = 0usize;
    let mut total_zero_advantages = 0usize;
    let mut max_abs_advantage = 0.0_f32;
    for trajectory in trajectories {
        *question_counts
            .entry(trajectory.question.flat_id.0)
            .or_default() += 1;
        if trajectory.input_ids.len() != trajectory.labels.len()
            || trajectory.input_ids.len() != trajectory.advantages.len()
        {
            invalid_shape_count += 1;
        }
        let trainable_tokens = trajectory
            .labels
            .iter()
            .filter(|&&item| item != -100)
            .count();
        if trainable_tokens == 0 {
            no_trainable_count += 1;
        }
        trainable_total += trainable_tokens;
        let (positive, negative, zero, min_advantage, max_advantage) =
            summarize_advantages(&trajectory.advantages);
        total_positive_advantages += positive;
        total_negative_advantages += negative;
        total_zero_advantages += zero;
        max_abs_advantage = max_abs_advantage
            .max(min_advantage.abs())
            .max(max_advantage.abs());
    }
    let mut per_question_counts = question_counts.values().copied().collect::<Vec<_>>();
    per_question_counts.sort_unstable();
    println!(
        "questions count={} per_question_min={} per_question_p50={} per_question_p90={} per_question_max={}",
        question_counts.len(),
        per_question_counts.first().copied().unwrap_or(0),
        percentile(&per_question_counts, 50, 100),
        percentile(&per_question_counts, 90, 100),
        per_question_counts.last().copied().unwrap_or(0),
    );
    println!(
        "validity invalid_shape={} no_trainable={} trainable_tokens={} adv_pos={} adv_neg={} adv_zero={} max_abs_adv={:.6}",
        invalid_shape_count,
        no_trainable_count,
        trainable_total,
        total_positive_advantages,
        total_negative_advantages,
        total_zero_advantages,
        max_abs_advantage,
    );
    if let Some(statistics) = statistics {
        println!(
            "stats total={} adopted={} cutoff={:.6} avg_abs_min={:.6} avg_abs_max={:.6} balance_pre_pos={:.6} balance_pre_neg={:.6} mult_pos={:.6} mult_neg={:.6} balance_post_pos={:.6} balance_post_neg={:.6}",
            statistics.total_trajectories,
            statistics.adopted_trajectories,
            statistics.average_absolute_advantage_cutoff,
            statistics.min_average_absolute_advantage,
            statistics.max_average_absolute_advantage,
            statistics.pre_balance_total_positive_advantage,
            statistics.pre_balance_total_negative_advantage_magnitude,
            statistics.positive_advantage_multiplier,
            statistics.negative_advantage_multiplier,
            statistics.post_balance_total_positive_advantage,
            statistics.post_balance_total_negative_advantage_magnitude,
        );
    }

    for prefix_len in [100usize, 1000, 3000, 6000, trajectories.len()] {
        if prefix_len > trajectories.len() {
            continue;
        }
        print_prefix_summary(prefix_len, &trajectories[..prefix_len]);
    }

    for index in 0..sample_count.min(trajectories.len()) {
        print_sample("first", index, &trajectories[index]);
    }
    let tail_start = trajectories.len().saturating_sub(sample_count);
    for index in tail_start..trajectories.len() {
        if index >= sample_count {
            print_sample("last", index, &trajectories[index]);
        }
    }

    if dump_decoded_samples {
        print_decoded_samples(
            trajectories,
            sample_count,
            sample_indices,
            decoded_char_limit,
        );
    }
}

fn print_decoded_samples<M: LlmModelMarker>(
    trajectories: &[DirectTrainingTrajectory<M>],
    sample_count: usize,
    sample_indices: &[usize],
    decoded_char_limit: usize,
) {
    let indices = if sample_indices.is_empty() {
        let mut indices = Vec::new();
        indices.extend(0..sample_count.min(trajectories.len()));
        let middle = trajectories.len() / 2;
        for offset in 0..sample_count.min(trajectories.len().saturating_sub(middle)) {
            indices.push(middle + offset);
        }
        let tail_start = trajectories.len().saturating_sub(sample_count);
        indices.extend(tail_start..trajectories.len());
        indices.sort_unstable();
        indices.dedup();
        indices
    } else {
        sample_indices
            .iter()
            .copied()
            .filter(|&index| index < trajectories.len())
            .collect::<Vec<_>>()
    };

    for index in indices {
        let trajectory = &trajectories[index];
        print_sample("decoded", index, trajectory);
        let decoded = <M::Tokenizer as MyTokenizer<M>>::decode_i32_ids(&trajectory.input_ids);
        let decoded_chars = decoded.chars().count();
        let truncated = if decoded_chars > decoded_char_limit {
            format!(
                "{}\n[truncated: {} chars total, showing first {}]",
                decoded.chars().take(decoded_char_limit).collect::<String>(),
                decoded_chars,
                decoded_char_limit,
            )
        } else {
            decoded
        };
        println!("decoded_begin index={}", index);
        println!("{}", truncated);
        println!("decoded_end index={}", index);
    }
}

fn print_prefix_summary<M: LlmModelMarker>(
    prefix_len: usize,
    trajectories: &[DirectTrainingTrajectory<M>],
) {
    let mut trainable_total = 0usize;
    let mut positive = 0usize;
    let mut negative = 0usize;
    let mut zero = 0usize;
    let mut lengths = trajectories
        .iter()
        .map(|trajectory| trajectory.input_ids.len())
        .collect::<Vec<_>>();
    lengths.sort_unstable();
    for trajectory in trajectories {
        trainable_total += trajectory
            .labels
            .iter()
            .filter(|&&item| item != -100)
            .count();
        let (pos, neg, zer, _, _) = summarize_advantages(&trajectory.advantages);
        positive += pos;
        negative += neg;
        zero += zer;
    }
    println!(
        "prefix count={} len_min={} len_p50={} len_max={} trainable_tokens={} adv_pos={} adv_neg={} adv_zero={} pos_neg_ratio={:.4}",
        prefix_len,
        lengths.first().copied().unwrap_or(0),
        percentile(&lengths, 50, 100),
        lengths.last().copied().unwrap_or(0),
        trainable_total,
        positive,
        negative,
        zero,
        if negative == 0 {
            f64::INFINITY
        } else {
            positive as f64 / negative as f64
        },
    );
}
