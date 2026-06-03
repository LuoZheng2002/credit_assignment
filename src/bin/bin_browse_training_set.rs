use std::backtrace::Backtrace;
use std::error::Error;
use std::io::{self, Stdout};
use std::sync::{Arc, Mutex};

use clap::Parser;
use credit_assignment::direct_tool::direct_rollout_config::AdvantageCalculationPolicy;
use credit_assignment::{
    direct_tool::{
        direct_rollout_config::DirectRolloutConfig,
        direct_training_set::{
            AssetFileTrainingTrajectories, DirectTrainingSetStatistics, DirectTrainingTrajectory,
        },
        posterior_calculation_config::{PosteriorCalculationConfig, PosteriorHyperparameters},
    },
    json_line_util::read_json,
    llm_model::{
        Gpt4o, LlmModelMarker, LlmModelName, MyTokenizer, Qwen3_4B, Qwen3_06B, Qwen25_7B,
        Qwen35_4B, Qwen35_08B,
    },
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
use research_utility::{asset_file::AssetFile, sqlite_store::SqliteStore};

#[derive(Parser, Debug)]
#[command(author, version, about = "Interactively browse training sets")]
struct Args {
    #[arg(value_enum, short, long)]
    model: LlmModelName,
    #[arg(long)]
    config_nickname: String,
    #[arg(long)]
    rollout_config_path: String,
    #[arg(long)]
    posterior_hyperparameters_path: String,
    #[arg(long)]
    epoch: usize, // the epoch index
    #[arg(long)]
    cumulative_avg_abs_advantage_cutoff: f32,
    #[arg(long, value_enum)]
    advantage_calculation_policy: AdvantageCalculationPolicy,
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
    store: Arc<Mutex<SqliteStore<usize, DirectTrainingTrajectory<M>>>>,
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
    loaded: Option<LoadedTrajectory<M>>,
    statistics: Option<DirectTrainingSetStatistics>,
    jump_input: String,
    jump_input_active: bool,
    jump_status: Option<String>,
}

impl<M: LlmModelMarker> App<M> {
    fn new(
        store: Arc<Mutex<SqliteStore<usize, DirectTrainingTrajectory<M>>>>,
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
            loaded: None,
            statistics,
            jump_input: String::new(),
            jump_input_active: false,
            jump_status: None,
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
            .lock()
            .unwrap()
            .get(key)
            .unwrap()
            .expect("key from sqlite key set must exist");

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
                loaded.trajectory.question.flat_id,
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
                    "\n\n[training set stats] total: {}  adopted: {}\nmax avg absolute advantage: {:.6}  cutoff: {:.6}  min avg absolute advantage: {:.6}",
                    statistics.total_trajectories,
                    statistics.adopted_trajectories,
                    statistics.max_average_absolute_advantage,
                    statistics.average_absolute_advantage_cutoff,
                    statistics.min_average_absolute_advantage,
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
            "Left/Right: prev/next trajectory  /: jump by key  Tab: switch focus  Up/Down/PgUp/PgDn/Home/End: scroll focused pane  Mouse wheel: scroll focused pane  q: quit",
        )
        .block(Block::default().borders(Borders::ALL).title("Controls"));
        frame.render_widget(controls, chunks[3]);
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

        match mouse.kind {
            MouseEventKind::ScrollUp => self.scroll_focused_up(1),
            MouseEventKind::ScrollDown => self.scroll_focused_down(1),
            _ => {}
        }
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
    rollout_config: DirectRolloutConfig,
    posterior_calculation_config: PosteriorCalculationConfig,
    epoch: usize, // the epoch index
    cumulative_avg_abs_advantage_cutoff: f32,
    advantage_calculation_policy: AdvantageCalculationPolicy,
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
        model,
        config_nickname,
        rollout_config_path,
        posterior_hyperparameters_path,
        epoch,
        cumulative_avg_abs_advantage_cutoff,
        advantage_calculation_policy,
    } = Args::parse();
    let rollout_config: DirectRolloutConfig = read_json(rollout_config_path).unwrap();
    let posterior_hyperparameters =
        read_json::<PosteriorHyperparameters>(posterior_hyperparameters_path).unwrap();
    let posterior_calculation_config = PosteriorCalculationConfig {
        hyperparameters: posterior_hyperparameters,
    };
    let run_program_args = RunProgramArgs {
        config_nickname,
        rollout_config,
        posterior_calculation_config,
        epoch,
        cumulative_avg_abs_advantage_cutoff,
        advantage_calculation_policy,
    };
    match model {
        LlmModelName::Gpt4o => run_program::<Gpt4o>(run_program_args).await,
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
        rollout_config,
        posterior_calculation_config,
        epoch,
        cumulative_avg_abs_advantage_cutoff,
        advantage_calculation_policy,
    } = run_program_args;
    let asset_file_training_set = AssetFileTrainingTrajectories::<M> {
        config_nickname: config_nickname.clone(),
        rollout_config,
        posterior_calculation_config,
        epoch,
        cumulative_avg_abs_advantage_cutoff,
        advantage_calculation_policy,
        _phantom: std::marker::PhantomData::<M>,
    };
    let training_set_store = asset_file_training_set.fetch().await;
    let training_set_store = Arc::new(Mutex::new(training_set_store));
    let mut keys = training_set_store.lock().unwrap().get_keys().unwrap();
    keys.sort();
    let statistics =
        read_json::<DirectTrainingSetStatistics>(asset_file_training_set.statistics_file_path())
            .ok();

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
