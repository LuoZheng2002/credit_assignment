use std::backtrace::Backtrace;
use std::error::Error;
use std::io::{self, Stdout};

use clap::Parser;
use credit_assignment::{
    direct_tool::{
        direct_rollout_config::DirectRolloutConfig,
        direct_training_set::{
            AssetFileTrainingTrajectories, DirectTrainingSetStatistics, DirectTrainingTrajectory,
        },
        posterior_calculation_config::{PosteriorCalculationConfig, PosteriorHyperparameters},
    },
    json_line_util::read_json,
    llm_model::{Gpt4o, Gpt5Mini, LlmModelMarker, LlmModelName, MyTokenizer, Qwen3_4B, Qwen25, Qwen35_4B},
};
use crossterm::event::{self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode, KeyEvent};
use crossterm::execute;
use crossterm::terminal::{
    EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
};
use ratatui::Terminal;
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Direction, Layout};
use ratatui::style::{Color, Style};
use ratatui::text::{Line, Span, Text};
use ratatui::widgets::{Block, Borders, Paragraph, Wrap};
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
    max_num_training_trajectories: usize,
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

struct App<M: LlmModelMarker> {
    store: SqliteStore<usize, DirectTrainingTrajectory<M>>,
    keys: Vec<usize>,
    selected_index: usize,
    conversation_scroll: usize,
    conversation_max_scroll: usize,
    loaded: Option<LoadedTrajectory<M>>,
    statistics: Option<DirectTrainingSetStatistics>,
}

impl<M: LlmModelMarker> App<M> {
    fn new(
        store: SqliteStore<usize, DirectTrainingTrajectory<M>>,
        keys: Vec<usize>,
        statistics: Option<DirectTrainingSetStatistics>,
    ) -> Self {
        Self {
            store,
            keys,
            selected_index: 0,
            conversation_scroll: 0,
            conversation_max_scroll: 0,
            loaded: None,
            statistics,
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
        let trajectory = tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(async {
                self.store
                    .get(key)
                    .await
                    .unwrap()
                    .expect("key from sqlite key set must exist")
            })
        });

        let conversation_render = build_conversation_render::<M>(&trajectory);
        self.loaded = Some(LoadedTrajectory {
            index: self.selected_index,
            key,
            trajectory,
            conversation_render,
        });
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
                "Model: {}\nTrajectory: {}/{} (key #{})\nQuestion #{} [{}:{}]\nAverage segment advantage: {:.6}\nToken count: {}  Trainable tokens: {}",
                M::CLI_NAME,
                loaded.index + 1,
                self.keys.len(),
                loaded.key,
                loaded.trajectory.question.flat_id,
                loaded.trajectory.question.dataset_name,
                loaded.trajectory.question.question_id,
                loaded.trajectory.average_segment_advantage,
                loaded.trajectory.input_ids.len(),
                trainable_tokens,
            );
            if let Some(statistics) = &self.statistics {
                text.push_str(&format!(
                    "\n\n[training set stats] total: {}  adopted: {}\nmax avg advantage: {:.6}  cutoff: {:.6}  min avg advantage: {:.6}",
                    statistics.total_trajectories,
                    statistics.adopted_trajectories,
                    statistics.max_average_advantage,
                    statistics.average_advantage_cutoff,
                    statistics.min_average_advantage,
                ));
            }
            text
        };

        frame.render_widget(
            Paragraph::new(summary_text)
                .wrap(Wrap { trim: false })
                .block(Block::default().borders(Borders::ALL).title("Metadata")),
            chunks[0],
        );

        if self.keys.is_empty() {
            frame.render_widget(
                Paragraph::new("No conversation to render")
                    .block(Block::default().borders(Borders::ALL).title("Conversation")),
                chunks[1],
            );
        } else {
            let loaded = self
                .loaded
                .as_ref()
                .expect("selected trajectory must be loaded");
            let line_count = loaded.conversation_render.plain.lines().count().max(1);
            let conversation_height = chunks[1].height.saturating_sub(2) as usize;
            self.conversation_max_scroll = line_count.saturating_sub(conversation_height);
            if self.conversation_scroll > self.conversation_max_scroll {
                self.conversation_scroll = self.conversation_max_scroll;
            }
            frame.render_widget(
                Paragraph::new(loaded.conversation_render.styled.clone())
                    .wrap(Wrap { trim: false })
                    .scroll((clamp_scroll(self.conversation_scroll), 0))
                    .block(Block::default().borders(Borders::ALL).title("Conversation")),
                chunks[1],
            );
        }

        let controls = Paragraph::new(
            "Left/Right: prev/next trajectory  Up/Down/PgUp/PgDn/Home/End: scroll conversation  q: quit",
        )
        .block(Block::default().borders(Borders::ALL).title("Controls"));
        frame.render_widget(controls, chunks[2]);
    }

    fn handle_key(&mut self, key: KeyEvent) -> bool {
        match key.code {
            KeyCode::Char('q') | KeyCode::Esc => true,
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
                self.conversation_scroll = self.conversation_scroll.saturating_sub(1);
                false
            }
            KeyCode::Down => {
                self.conversation_scroll = self.conversation_scroll.saturating_add(1);
                false
            }
            KeyCode::PageUp => {
                self.conversation_scroll = self.conversation_scroll.saturating_sub(10);
                false
            }
            KeyCode::PageDown => {
                self.conversation_scroll = self.conversation_scroll.saturating_add(10);
                false
            }
            KeyCode::Home => {
                self.conversation_scroll = 0;
                false
            }
            KeyCode::End => {
                self.conversation_scroll = self.conversation_max_scroll;
                false
            }
            _ => false,
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

fn advantage_to_color(advantage: f32) -> Color {
    let x = (advantage.clamp(-3.0, 3.0)) / 3.0;
    let (red, green) = if x >= 0.0 {
        (((1.0 - x) * 255.0).round() as u8, 255)
    } else {
        (255, ((1.0 + x) * 255.0).round() as u8)
    };
    Color::Rgb(red, green, 0)
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
            lines.last_mut()
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
            _ => {}
        }
    }
    Ok(())
}

#[tokio::main]
async fn main() {
    std::panic::set_hook(Box::new(|info| {
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
        max_num_training_trajectories,
    } = Args::parse();
    let rollout_config: DirectRolloutConfig = read_json(rollout_config_path).unwrap();
    let posterior_hyperparameters =
        read_json::<PosteriorHyperparameters>(posterior_hyperparameters_path).unwrap();
    let posterior_calculation_config = PosteriorCalculationConfig {
        hyperparameters: posterior_hyperparameters,
    };
    match model {
        LlmModelName::Gpt4o => {
            run_program::<Gpt4o>(
                config_nickname,
                rollout_config,
                posterior_calculation_config,
                max_num_training_trajectories,
            )
            .await
        }
        LlmModelName::Gpt5Mini => {
            run_program::<Gpt5Mini>(
                config_nickname,
                rollout_config,
                posterior_calculation_config,
                max_num_training_trajectories,
            )
            .await
        }
        LlmModelName::Qwen3_4b => {
            run_program::<Qwen3_4B>(
                config_nickname,
                rollout_config,
                posterior_calculation_config,
                max_num_training_trajectories,
            )
            .await
        }
        LlmModelName::Qwen25_7b => {
            run_program::<Qwen25>(
                config_nickname,
                rollout_config,
                posterior_calculation_config,
                max_num_training_trajectories,
            )
            .await
        }
        LlmModelName::Qwen35_4b => {
            run_program::<Qwen35_4B>(
                config_nickname,
                rollout_config,
                posterior_calculation_config,
                max_num_training_trajectories,
            )
            .await
        }
    }
}

pub async fn run_program<M: LlmModelMarker>(
    config_nickname: String,
    rollout_config: DirectRolloutConfig,
    posterior_calculation_config: PosteriorCalculationConfig,
    max_num_training_trajectories: usize,
) {
    let asset_file_training_set = AssetFileTrainingTrajectories::<M> {
        config_nickname: config_nickname.clone(),
        rollout_config,
        posterior_calculation_config,
        max_num_training_trajectories,
        _phantom: std::marker::PhantomData::<M>,
    };
    let training_set_store = asset_file_training_set.fetch().await;
    let mut keys = training_set_store.get_keys().await.unwrap();
    keys.sort();
    let statistics = read_json::<DirectTrainingSetStatistics>(asset_file_training_set.statistics_file_path()).ok();

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
