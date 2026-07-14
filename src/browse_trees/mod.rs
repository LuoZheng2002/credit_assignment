mod context;
mod tree_render;

use std::error::Error;
use std::io::{self, Stdout};
use std::marker::PhantomData;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use ordered_float::NotNan;

use crate::config_paths::{ConfigPaths, config_paths_file_path_from_action_logs_path};
use crate::hybrid_dataset::{
    DatasetSplit, DatasetSplitEnum, HybridDatasetStore, QuestionFlatId, Testing, Training,
    Validation, open_hybrid_dataset,
};
use crate::json_toml_utils::read_json;
use crate::llm_model::{
    Gemma3_4BIt, Llama31_8BInstruct, LlmModelMarker, LlmModelName, Mistral7BInstructV03, Qwen3_4B,
    Qwen3_06B, Qwen25_7B, Qwen35_4B, Qwen35_08B,
};
use crate::{
    constants,
    posterior_calculation_config::{PosteriorCalculationConfig, PosteriorHyperparameters},
    rollout_config::RolloutConfig,
    tree_action_log::{
        ActionLogConfigBundle, ActionLogStore, DirectTreeActionLog,
        action_log_config_bundle_file_path,
    },
};
use crossterm::cursor::Show;
use crossterm::event::{
    self, EnableMouseCapture, Event, KeyCode, KeyEvent, MouseButton, MouseEvent, MouseEventKind,
};
use crossterm::execute;
use crossterm::terminal::{
    EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
};
use ratatui::Terminal;
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::widgets::{Block, Borders, List, ListItem, ListState, Paragraph, Wrap};

use tokio::sync::mpsc::{UnboundedReceiver, UnboundedSender, unbounded_channel};

use self::context::parse_action_logs_context;
use self::tree_render::*;

const TREE_MOUSE_SCROLL_DEBOUNCE: Duration = Duration::from_millis(50);

struct HomePageLoadRequest {
    page_start: usize,
    started_at: Instant,
}

struct App<M: LlmModelMarker, S: DatasetSplit> {
    _model_marker: PhantomData<M>,
    override_hyperparameters: Option<PosteriorHyperparameters>,
    rollout_config: RolloutConfig<S>,
    posterior_calculation_config: PosteriorCalculationConfig,
    use_tool: bool,
    fixed_temperature: NotNan<f32>,
    question_store: HybridDatasetStore<S>,
    action_store: ActionLogStore<M, S>,
    entry_keys: Vec<QuestionFlatId<S>>,
    entry_cache: Vec<EntryLoadState<M, S>>,
    entry_load_tx: UnboundedSender<EntryLoadResult<M, S>>,
    entry_load_rx: UnboundedReceiver<EntryLoadResult<M, S>>,
    mode: Mode,
    home_selected_index: usize,
    home_list_area: Option<Rect>,
    summary_area: Option<Rect>,
    conversation_area: Option<Rect>,
    tree_page: Option<TreePage<M, S>>,
    tree_focus: TreePaneFocus,
    tree_scroll_mode: TreeScrollMode,
    tree_color_mode: TreeColorMode,
    tree_horizontal_scroll: usize,
    summary_scroll: usize,
    summary_max_scroll: usize,
    summary_metrics: Option<PaneMetrics>,
    conversation_scroll: usize,
    conversation_max_scroll: usize,
    conversation_metrics: Option<PaneMetrics>,
    home_page_load_request: Option<HomePageLoadRequest>,
    last_tree_scroll_at: Option<Instant>,
}

impl<M: LlmModelMarker, S: DatasetSplit> App<M, S> {
    async fn new(
        _config_nickname: String,
        rollout_config: RolloutConfig<S>,
        posterior_calculation_config: PosteriorCalculationConfig,
        use_tool: bool,
        fixed_temperature: NotNan<f32>,
        _epoch: usize,
        // entry_keys: Vec<usize>,
        override_hyperparameters: Option<PosteriorHyperparameters>,
        action_logs_path: PathBuf,
    ) -> Self {
        let (entry_load_tx, entry_load_rx) = unbounded_channel();
        let question_store = open_hybrid_dataset::<S>();
        let action_store = ActionLogStore::<M, S>::initialize_if_missing(&action_logs_path)
            .unwrap_or_else(|e| {
                panic!(
                    "Failed to open direct action log store at {}: {}",
                    action_logs_path.display(),
                    e
                )
            });
        action_store.sort().unwrap();
        let mut entry_keys = action_store.get_keys().unwrap();
        entry_keys.sort_by_key(|key| key.0);
        let entry_cache = std::iter::repeat_with(|| EntryLoadState::Unloaded)
            .take(entry_keys.len())
            .collect();
        Self {
            _model_marker: PhantomData,
            override_hyperparameters,
            rollout_config,
            posterior_calculation_config,
            use_tool,
            fixed_temperature,
            entry_keys,
            entry_cache,
            entry_load_tx,
            entry_load_rx,
            mode: Mode::Home,
            home_selected_index: 0,
            home_list_area: None,
            summary_area: None,
            conversation_area: None,
            tree_page: None,
            tree_focus: TreePaneFocus::Tree,
            tree_scroll_mode: TreeScrollMode::Scaling,
            tree_color_mode: TreeColorMode::SignalToNoise,
            tree_horizontal_scroll: 0,
            summary_scroll: 0,
            summary_max_scroll: 0,
            summary_metrics: None,
            conversation_scroll: 0,
            conversation_max_scroll: 0,
            conversation_metrics: None,
            home_page_load_request: None,
            last_tree_scroll_at: None,
            question_store,
            action_store,
        }
    }

    fn total_entries(&self) -> usize {
        self.entry_keys.len()
    }

    fn poll_entry_load_results(&mut self) {
        while let Ok(result) = self.entry_load_rx.try_recv() {
            if result.index < self.total_entries() {
                self.entry_cache[result.index] = result.state;
            }
        }
    }

    fn request_entry_load_if_needed(&mut self, index: usize) {
        if index >= self.total_entries() {
            return;
        }
        match self.entry_cache.get(index) {
            Some(EntryLoadState::Loaded(_)) | Some(EntryLoadState::Loading) => return,
            Some(EntryLoadState::Failed(_)) => return,
            Some(EntryLoadState::Unloaded) => {}
            None => return,
        }
        self.entry_cache[index] = EntryLoadState::Loading;
        let key = self.entry_keys[index];
        let state = match self.action_store.load_action_log(key) {
            Ok(actions) => {
                let question = self.question_store.get(key).unwrap().unwrap();
                let action_log = DirectTreeActionLog {
                    question,
                    rollout_config: self.rollout_config.clone(),
                    posterior_calculation_config: self.posterior_calculation_config.clone(),
                    use_tool: self.use_tool,
                    fixed_temperature: self.fixed_temperature,
                    actions,
                };
                let (num_correct, num_leaves, win_rate) =
                    question_stats_from_action_log::<M, S>(&action_log);
                EntryLoadState::Loaded(QuestionEntry {
                    key,
                    action_log,
                    win_rate,
                    num_correct,
                    num_leaves,
                })
            }
            Err(error) => EntryLoadState::Failed(format!(
                "Failed to load question key {} from action log store: {}",
                key.0, error
            )),
        };
        let _ = self.entry_load_tx.send(EntryLoadResult { index, state });
    }

    fn schedule_home_page_load(&mut self, page_start: usize) {
        match self.home_page_load_request {
            Some(ref request) if request.page_start == page_start => {}
            _ => {
                self.home_page_load_request = Some(HomePageLoadRequest {
                    page_start,
                    started_at: Instant::now(),
                });
            }
        }
    }

    fn maybe_load_home_page(&mut self, page_start: usize) {
        let should_load = self.home_page_load_request.as_ref().is_some_and(|request| {
            request.page_start == page_start
                && request.started_at.elapsed() >= Duration::from_millis(100)
        });
        if should_load {
            self.load_visible_home_page(page_start);
            self.home_page_load_request = None;
        }
    }

    fn load_visible_home_page(&mut self, page_start: usize) {
        if self.total_entries() == 0 {
            return;
        }
        let page_end = (page_start + QUESTIONS_PER_PAGE).min(self.total_entries());
        for index in page_start..page_end {
            self.request_entry_load_if_needed(index);
        }
    }

    fn loaded_entry(&self, index: usize) -> Option<&QuestionEntry<M, S>> {
        let Some(state) = self.entry_cache.get(index) else {
            return None;
        };
        let EntryLoadState::Loaded(entry) = state else {
            return None;
        };
        Some(entry)
    }

    fn draw(&mut self, frame: &mut ratatui::Frame<'_>) {
        self.poll_entry_load_results();
        match self.mode {
            Mode::Home => self.draw_home(frame),
            Mode::Tree => self.draw_tree(frame),
        }
    }

    fn handle_key(&mut self, key: KeyEvent) -> bool {
        match self.mode {
            Mode::Home => self.handle_home_key(key),
            Mode::Tree => self.handle_tree_key(key),
        }
    }

    fn handle_mouse(&mut self, mouse: MouseEvent) {
        match self.mode {
            Mode::Home => self.handle_home_mouse(mouse),
            Mode::Tree => self.handle_tree_mouse(mouse),
        }
    }

    fn draw_home(&mut self, frame: &mut ratatui::Frame<'_>) {
        self.summary_area = None;
        self.conversation_area = None;
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(3),
                Constraint::Min(5),
                Constraint::Length(3),
            ])
            .split(frame.area());
        self.home_list_area = Some(chunks[1]);

        let total_pages = if self.total_entries() == 0 {
            1
        } else {
            self.total_entries().div_ceil(QUESTIONS_PER_PAGE)
        };
        let current_page = if self.total_entries() == 0 {
            1
        } else {
            self.home_selected_index / QUESTIONS_PER_PAGE + 1
        };

        let title = Paragraph::new(format!(
            "Questions and win rate (page {current_page}/{total_pages}, total {})",
            self.total_entries()
        ))
        .block(Block::default().borders(Borders::ALL).title("Home"));
        frame.render_widget(title, chunks[0]);

        if self.total_entries() == 0 {
            let empty = Paragraph::new("No direct session logs found")
                .block(Block::default().borders(Borders::ALL).title("Questions"));
            frame.render_widget(empty, chunks[1]);
        } else {
            let page_start = (self.home_selected_index / QUESTIONS_PER_PAGE) * QUESTIONS_PER_PAGE;
            self.schedule_home_page_load(page_start);
            self.maybe_load_home_page(page_start);
            let page_end = (page_start + QUESTIONS_PER_PAGE).min(self.total_entries());
            let items: Vec<ListItem> = (page_start..page_end)
                .map(|index| {
                    let key = self.entry_keys[index];
                    let text = match self.entry_cache.get(index) {
                        Some(EntryLoadState::Loaded(entry)) => {
                            let question_preview =
                                single_line_preview(&entry.action_log.question.question, 72);
                            format!(
                                "#{}  win {:>5.1}% ({}/{})  [{}] {}",
                                entry.key.0,
                                entry.win_rate * 100.0,
                                entry.num_correct,
                                entry.num_leaves,
                                entry.action_log.question.dataset_name,
                                question_preview
                            )
                        }
                        Some(EntryLoadState::Loading) | Some(EntryLoadState::Unloaded) => {
                            format!("#{}  loading...", key.0)
                        }
                        Some(EntryLoadState::Failed(error)) => {
                            let preview = single_line_preview(error, 60);
                            format!("#{}  failed: {}", key.0, preview)
                        }
                        None => format!("#{}  unavailable", key.0),
                    };
                    ListItem::new(text)
                })
                .collect();
            let mut state = ListState::default();
            state.select(Some(self.home_selected_index - page_start));

            frame.render_stateful_widget(
                List::new(items)
                    .block(Block::default().borders(Borders::ALL).title("Questions"))
                    .highlight_style(
                        Style::default()
                            .fg(Color::Cyan)
                            .add_modifier(Modifier::BOLD),
                    )
                    .highlight_symbol("> "),
                chunks[1],
                &mut state,
            );
        }

        let controls = Paragraph::new(
            "Up/Down: select question  Left/Right: prev/next page  Enter: open tree  q: quit",
        )
        .block(Block::default().borders(Borders::ALL).title("Controls"));
        frame.render_widget(controls, chunks[2]);
    }

    fn draw_tree(&mut self, frame: &mut ratatui::Frame<'_>) {
        let Some(entry_index) = self.tree_page.as_ref().map(|page| page.entry_index) else {
            return;
        };
        let Some(entry) = self.loaded_entry(entry_index).cloned() else {
            let loading = Paragraph::new("Loading selected question...")
                .block(Block::default().borders(Borders::ALL).title("Tree"));
            frame.render_widget(loading, frame.area());
            return;
        };

        let Some(tree_page) = self.tree_page.as_mut() else {
            return;
        };
        let selected_segment = tree_page
            .tree_snapshot
            .segments
            .get(&tree_page.selected_segment_id)
            .expect("selected segment must exist");

        let tree_window_height = (tree_page.tree_lines.len() + 2).max(6);
        let tree_window_height_u16 = if tree_window_height > u16::MAX as usize {
            u16::MAX
        } else {
            tree_window_height as u16
        };

        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(8),
                Constraint::Min(8),
                Constraint::Length(tree_window_height_u16),
            ])
            .split(frame.area());

        let judgment = tree_page
            .tree_snapshot
            .leaf_segment_judgments
            .get(&tree_page.selected_segment_id);
        let posterior_stats = tree_page
            .segment_posterior_stats
            .get(&tree_page.selected_segment_id)
            .copied();

        let posterior_mean_text = posterior_stats
            .map(|stats| format!("{:.6}", stats.posterior_mean))
            .unwrap_or_else(|| "N/A".to_string());
        let posterior_std_text = posterior_stats
            .map(|stats| format!("{:.6}", stats.posterior_std))
            .unwrap_or_else(|| "N/A".to_string());
        let signal_to_noise_text = posterior_stats
            .map(|stats| format!("{:.6}", stats.signal_to_noise))
            .unwrap_or_else(|| "N/A".to_string());
        let selected_advantage_from_posterior = tree_page
            .segment_advantages_from_posteriors
            .get(&tree_page.selected_segment_id)
            .copied()
            .unwrap_or(0.0);
        let selected_advantage_from_win_rate = tree_page
            .segment_advantages_from_win_rate
            .get(&tree_page.selected_segment_id)
            .copied()
            .unwrap_or(0.0);
        let selected_trajectory_length_tokens = tree_page
            .tree_snapshot
            .get_trajectory_length_till_id(tree_page.selected_segment_id);

        let mut summary = format!(
            "Question #{}\nQuestion: {}\nCorrect answer: {}\nActions applied: {}/{}\nSelected segment: S{} (children: {})\nTrajectory length (tokens): {}\nposterior_mean: {}\nposterior_std: {}\nsignal_to_noise: {}\nadvantage_from_posterior: {:.6}\nadvantage_from_win_rate: {:.6}",
            entry.key.0,
            entry.action_log.question.question,
            entry.action_log.question.correct_answer,
            tree_page.action_limit,
            tree_page.total_actions,
            tree_page.selected_segment_id.0,
            selected_segment.child_ids.len(),
            selected_trajectory_length_tokens,
            posterior_mean_text,
            posterior_std_text,
            signal_to_noise_text,
            selected_advantage_from_posterior,
            selected_advantage_from_win_rate
        );
        match self.tree_color_mode {
            TreeColorMode::SignalToNoise => {
                if let Some((min_value, max_value)) =
                    posterior_stat_min_max(&tree_page.segment_posterior_stats, |stats| {
                        stats.signal_to_noise
                    })
                {
                    summary.push_str(&format!(
                        "\n[signal_to_noise range] min: {:.6}, max: {:.6}",
                        min_value, max_value
                    ));
                }
            }
            TreeColorMode::PosteriorMean => {
                if let Some((min_value, max_value)) =
                    posterior_stat_min_max(&tree_page.segment_posterior_stats, |stats| {
                        stats.posterior_mean
                    })
                {
                    summary.push_str(&format!(
                        "\n[posterior_mean range] min: {:.6}, max: {:.6}",
                        min_value, max_value
                    ));
                }
            }
            TreeColorMode::PosteriorStd => {
                if let Some((min_value, max_value)) =
                    posterior_stat_min_max(&tree_page.segment_posterior_stats, |stats| {
                        stats.posterior_std
                    })
                {
                    summary.push_str(&format!(
                        "\n[posterior_std range] min: {:.6}, max: {:.6}",
                        min_value, max_value
                    ));
                }
            }
            TreeColorMode::BranchingScore => {}
            TreeColorMode::AdvantageFromPosterior => {
                summary.push_str(
                    "\n[advantage_from_posterior range] min: -3.000000, max: 3.000000 (clamped)",
                );
            }
            TreeColorMode::AdvantageFromWinRate => {
                summary.push_str(
                    "\n[advantage_from_win_rate range] min: -3.000000, max: 3.000000 (clamped)",
                );
            }
        }
        if let Some(judgment) = judgment {
            let model_answer = judgment.model_answer.model_answer_text();
            summary.push_str(&format!(
                "\nLeaf judgment: {}\nModel answer: {}",
                if judgment.is_correct {
                    "CORRECT"
                } else {
                    "WRONG"
                },
                model_answer
            ));
        }

        self.summary_area = Some(chunks[0]);
        let summary_block = Block::default()
            .borders(Borders::ALL)
            .title("Summary")
            .border_style(if self.tree_focus == TreePaneFocus::Summary {
                Style::default()
                    .fg(Color::LightGreen)
                    .add_modifier(Modifier::BOLD)
            } else {
                Style::default()
            });
        let summary_inner = summary_block.inner(chunks[0]);
        let summary_height = summary_inner.height as usize;
        let summary_lines =
            compute_wrapped_line_count(&summary, summary_inner, &mut self.summary_metrics);
        self.summary_max_scroll = bottom_scroll_limit(summary_lines, summary_height.max(1));
        frame.render_widget(
            Paragraph::new(summary)
                .wrap(Wrap { trim: false })
                .block(summary_block)
                .scroll((clamp_scroll(self.summary_scroll), 0)),
            chunks[0],
        );

        let conversation_render = build_conversation_render(
            &tree_page.tree_snapshot,
            &tree_page.segment_posterior_signal_scaled,
            tree_page.selected_segment_id,
        );
        self.conversation_area = Some(chunks[1]);
        let conversation_block = Block::default()
            .borders(Borders::ALL)
            .title("Conversation up to selected segment")
            .border_style(if self.tree_focus == TreePaneFocus::Conversation {
                Style::default()
                    .fg(Color::LightGreen)
                    .add_modifier(Modifier::BOLD)
            } else {
                Style::default()
            });
        let conversation_inner = conversation_block.inner(chunks[1]);
        let conversation_height = conversation_inner.height as usize;
        let conversation_lines = compute_wrapped_line_count(
            &conversation_render.plain,
            conversation_inner,
            &mut self.conversation_metrics,
        );
        let conversation_max_scroll =
            bottom_scroll_limit(conversation_lines, conversation_height.max(1));
        self.conversation_max_scroll = conversation_max_scroll;
        frame.render_widget(
            Paragraph::new(conversation_render.styled)
                .wrap(Wrap { trim: false })
                .block(conversation_block)
                .scroll((clamp_scroll(self.conversation_scroll), 0)),
            chunks[1],
        );

        tree_page.tree_area = Some(chunks[2]);
        let list_items: Vec<ListItem> = (0..tree_page.tree_lines.len())
            .map(|row| {
                ListItem::new(render_tree_line(
                    tree_page,
                    row,
                    self.tree_horizontal_scroll,
                    self.tree_color_mode,
                ))
            })
            .collect();
        let mut state = ListState::default();
        state.select(None);
        let tree_block = Block::default()
            .borders(Borders::ALL)
            .title(
                format!(
                    "Segment tree [scroll:{} color:{} ratio:{} hscroll:{} actions:{}/{}] (1:scale 2:pan 3:evolve, 4:snr 5:branch 6:mean 7:std 8:advantage_from_posterior 9:advantage_from_win_rate, wheel follows scroll mode)",
                    self.tree_scroll_mode.label(),
                    self.tree_color_mode.label(),
                    tree_page.width_division_ratio,
                    self.tree_horizontal_scroll,
                    tree_page.action_limit,
                    tree_page.total_actions,
                ),
            )
            .border_style(if self.tree_focus == TreePaneFocus::Tree {
                Style::default()
                    .fg(Color::LightGreen)
                    .add_modifier(Modifier::BOLD)
            } else {
                Style::default()
            });
        frame.render_stateful_widget(
            List::new(list_items)
                .block(tree_block)
                .highlight_style(Style::default())
                .highlight_symbol(""),
            chunks[2],
            &mut state,
        );
    }

    fn handle_home_key(&mut self, key: KeyEvent) -> bool {
        match key.code {
            KeyCode::Char('q') | KeyCode::Esc => true,
            KeyCode::Up => {
                self.home_selected_index = self.home_selected_index.saturating_sub(1);
                false
            }
            KeyCode::Down => {
                if self.total_entries() > 0 {
                    self.home_selected_index =
                        (self.home_selected_index + 1).min(self.total_entries() - 1);
                }
                false
            }
            KeyCode::Left => {
                self.home_selected_index =
                    self.home_selected_index.saturating_sub(QUESTIONS_PER_PAGE);
                false
            }
            KeyCode::Right => {
                if self.total_entries() > 0 {
                    self.home_selected_index = (self.home_selected_index + QUESTIONS_PER_PAGE)
                        .min(self.total_entries() - 1);
                }
                false
            }
            KeyCode::Enter => {
                self.open_selected_home_entry();
                false
            }
            _ => false,
        }
    }

    fn handle_tree_key(&mut self, key: KeyEvent) -> bool {
        if self.tree_page.is_none() {
            return false;
        }
        match key.code {
            KeyCode::Char('q') | KeyCode::Esc => {
                self.mode = Mode::Home;
                self.tree_page = None;
                self.tree_scroll_mode = TreeScrollMode::Scaling;
                self.tree_color_mode = TreeColorMode::SignalToNoise;
                self.tree_horizontal_scroll = 0;
                self.summary_area = None;
                self.summary_scroll = 0;
                self.summary_max_scroll = 0;
                self.summary_metrics = None;
                self.conversation_area = None;
                self.conversation_scroll = 0;
                self.conversation_max_scroll = 0;
                self.conversation_metrics = None;
                false
            }
            KeyCode::Left => {
                self.tree_horizontal_scroll = self.tree_horizontal_scroll.saturating_sub(1);
                false
            }
            KeyCode::Right => {
                self.tree_horizontal_scroll = self.tree_horizontal_scroll.saturating_add(1);
                false
            }
            KeyCode::Up => {
                match self.tree_focus {
                    TreePaneFocus::Summary => self.scroll_summary_up(1),
                    TreePaneFocus::Conversation => self.scroll_conversation_up(1),
                    TreePaneFocus::Tree => {}
                }
                false
            }
            KeyCode::Down => {
                match self.tree_focus {
                    TreePaneFocus::Summary => {
                        self.summary_scroll = self.summary_scroll.saturating_add(1)
                    }
                    TreePaneFocus::Conversation => self.scroll_conversation_down(1),
                    TreePaneFocus::Tree => {}
                }
                false
            }
            KeyCode::PageUp => {
                match self.tree_focus {
                    TreePaneFocus::Summary => self.scroll_summary_up(10),
                    TreePaneFocus::Conversation => self.scroll_conversation_up(10),
                    TreePaneFocus::Tree => {}
                }
                false
            }
            KeyCode::PageDown => {
                match self.tree_focus {
                    TreePaneFocus::Summary => {
                        self.summary_scroll = self.summary_scroll.saturating_add(10)
                    }
                    TreePaneFocus::Conversation => self.scroll_conversation_down(10),
                    TreePaneFocus::Tree => {}
                }
                false
            }
            KeyCode::Home => {
                match self.tree_focus {
                    TreePaneFocus::Summary => self.summary_scroll = 0,
                    TreePaneFocus::Conversation => self.conversation_scroll = 0,
                    TreePaneFocus::Tree => {}
                }
                false
            }
            KeyCode::Tab => {
                self.tree_focus = match self.tree_focus {
                    TreePaneFocus::Summary => TreePaneFocus::Conversation,
                    TreePaneFocus::Conversation => TreePaneFocus::Tree,
                    TreePaneFocus::Tree => TreePaneFocus::Summary,
                };
                false
            }
            KeyCode::Char('1') => {
                self.tree_scroll_mode = TreeScrollMode::Scaling;
                false
            }
            KeyCode::Char('2') => {
                self.tree_scroll_mode = TreeScrollMode::Panning;
                false
            }
            KeyCode::Char('3') => {
                self.tree_scroll_mode = TreeScrollMode::Evolution;
                false
            }
            KeyCode::Char('4') => {
                self.tree_color_mode = TreeColorMode::SignalToNoise;
                false
            }
            KeyCode::Char('5') => {
                self.tree_color_mode = TreeColorMode::BranchingScore;
                false
            }
            KeyCode::Char('6') => {
                self.tree_color_mode = TreeColorMode::PosteriorMean;
                false
            }
            KeyCode::Char('7') => {
                self.tree_color_mode = TreeColorMode::PosteriorStd;
                false
            }
            KeyCode::Char('8') => {
                self.tree_color_mode = TreeColorMode::AdvantageFromPosterior;
                false
            }
            KeyCode::Char('9') => {
                self.tree_color_mode = TreeColorMode::AdvantageFromWinRate;
                false
            }
            _ => false,
        }
    }

    fn handle_home_mouse(&mut self, mouse: MouseEvent) {
        let row_index = self.home_index_from_mouse(mouse.column, mouse.row);
        match mouse.kind {
            MouseEventKind::Moved => {
                if let Some(index) = row_index {
                    self.home_selected_index = index;
                }
            }
            MouseEventKind::Down(MouseButton::Left) => {
                if let Some(index) = row_index {
                    self.home_selected_index = index;
                    self.open_selected_home_entry();
                }
            }
            _ => {}
        }
    }

    fn should_process_tree_scroll(&mut self) -> bool {
        let now = Instant::now();
        if self
            .last_tree_scroll_at
            .is_some_and(|last| now.saturating_duration_since(last) < TREE_MOUSE_SCROLL_DEBOUNCE)
        {
            return false;
        }
        self.last_tree_scroll_at = Some(now);
        true
    }

    fn handle_tree_mouse(&mut self, mouse: MouseEvent) {
        if self.tree_page.is_none() {
            return;
        }
        if let Some(summary_area) = self.summary_area {
            if contains_point(summary_area, mouse.column, mouse.row) {
                self.tree_focus = TreePaneFocus::Summary;
            } else if let Some(conversation_area) = self.conversation_area {
                if contains_point(conversation_area, mouse.column, mouse.row) {
                    self.tree_focus = TreePaneFocus::Conversation;
                } else if let Some(tree_area) =
                    self.tree_page.as_ref().and_then(|page| page.tree_area)
                {
                    if contains_point(tree_area, mouse.column, mouse.row) {
                        self.tree_focus = TreePaneFocus::Tree;
                    }
                }
            }
        }

        let hovered = {
            let tree_page = self
                .tree_page
                .as_ref()
                .expect("tree page must exist in tree mode");
            tree_segment_at_mouse(
                tree_page,
                mouse.column,
                mouse.row,
                self.tree_horizontal_scroll,
            )
        };
        match mouse.kind {
            MouseEventKind::Moved => {
                if let Some(tree_page) = self.tree_page.as_mut() {
                    tree_page.hovered_segment_id = hovered;
                }
            }
            MouseEventKind::Down(MouseButton::Left) => {
                if let Some(tree_page) = self.tree_page.as_mut() {
                    tree_page.hovered_segment_id = hovered;
                    if let Some(segment_id) = hovered {
                        tree_page.selected_segment_id = segment_id;
                    }
                }
            }
            MouseEventKind::ScrollUp => {
                if !self.should_process_tree_scroll() {
                    return;
                }
                match self.tree_focus {
                    TreePaneFocus::Summary => self.scroll_summary_up(1),
                    TreePaneFocus::Conversation => self.scroll_conversation_up(1),
                    TreePaneFocus::Tree => {
                        let entry_index = self
                            .tree_page
                            .as_ref()
                            .expect("tree page must exist in tree mode")
                            .entry_index;
                        let Some(entry) = self.loaded_entry(entry_index).cloned() else {
                            return;
                        };
                        let tree_page = self
                            .tree_page
                            .as_mut()
                            .expect("tree page must exist in tree mode");
                        match self.tree_scroll_mode {
                            TreeScrollMode::Scaling => {
                                let old_ratio = tree_page.width_division_ratio;
                                let new_ratio = old_ratio.saturating_sub(1).max(1);
                                if old_ratio != new_ratio {
                                    self.tree_horizontal_scroll = scale_horizontal_scroll(
                                        self.tree_horizontal_scroll,
                                        old_ratio,
                                        new_ratio,
                                    );
                                    tree_page.set_width_division_ratio(
                                        &entry,
                                        new_ratio,
                                        self.override_hyperparameters.as_ref(),
                                    );
                                }
                            }
                            TreeScrollMode::Panning => {
                                self.tree_horizontal_scroll =
                                    self.tree_horizontal_scroll.saturating_sub(4);
                            }
                            TreeScrollMode::Evolution => {
                                let next = tree_page.action_limit.saturating_sub(1);
                                tree_page.set_action_limit(
                                    &entry,
                                    next,
                                    self.override_hyperparameters.as_ref(),
                                );
                            }
                        }
                    }
                }
            }
            MouseEventKind::ScrollDown => {
                if !self.should_process_tree_scroll() {
                    return;
                }
                match self.tree_focus {
                    TreePaneFocus::Summary => {
                        self.summary_scroll = self.summary_scroll.saturating_add(1)
                    }
                    TreePaneFocus::Conversation => self.scroll_conversation_down(1),
                    TreePaneFocus::Tree => {
                        let entry_index = self
                            .tree_page
                            .as_ref()
                            .expect("tree page must exist in tree mode")
                            .entry_index;
                        let Some(entry) = self.loaded_entry(entry_index).cloned() else {
                            return;
                        };
                        let tree_page = self
                            .tree_page
                            .as_mut()
                            .expect("tree page must exist in tree mode");
                        match self.tree_scroll_mode {
                            TreeScrollMode::Scaling => {
                                let old_ratio = tree_page.width_division_ratio;
                                let new_ratio = old_ratio.saturating_add(1);
                                self.tree_horizontal_scroll = scale_horizontal_scroll(
                                    self.tree_horizontal_scroll,
                                    old_ratio,
                                    new_ratio,
                                );
                                tree_page.set_width_division_ratio(
                                    &entry,
                                    new_ratio,
                                    self.override_hyperparameters.as_ref(),
                                );
                            }
                            TreeScrollMode::Panning => {
                                self.tree_horizontal_scroll =
                                    self.tree_horizontal_scroll.saturating_add(4);
                            }
                            TreeScrollMode::Evolution => {
                                let next =
                                    (tree_page.action_limit + 1).min(tree_page.total_actions);
                                tree_page.set_action_limit(
                                    &entry,
                                    next,
                                    self.override_hyperparameters.as_ref(),
                                );
                            }
                        }
                    }
                }
            }
            _ => {}
        }
    }

    fn open_selected_home_entry(&mut self) {
        if self.total_entries() == 0 {
            return;
        }
        self.request_entry_load_if_needed(self.home_selected_index);
        let Some(entry) = self.loaded_entry(self.home_selected_index).cloned() else {
            return;
        };
        self.tree_page = Some(TreePage::new(
            self.home_selected_index,
            &entry,
            self.override_hyperparameters.as_ref(),
        ));
        self.mode = Mode::Tree;
        self.tree_focus = TreePaneFocus::Tree;
        self.tree_scroll_mode = TreeScrollMode::Scaling;
        self.tree_color_mode = TreeColorMode::SignalToNoise;
        self.tree_horizontal_scroll = 0;
        self.summary_area = None;
        self.summary_scroll = 0;
        self.summary_max_scroll = 0;
        self.summary_metrics = None;
        self.conversation_scroll = 0;
        self.conversation_max_scroll = 0;
        self.conversation_metrics = None;
        self.conversation_area = None;
    }

    fn scroll_summary_up(&mut self, magnitude: usize) {
        if self.summary_scroll > self.summary_max_scroll {
            self.summary_scroll = self.summary_max_scroll;
            return;
        }
        self.summary_scroll = self.summary_scroll.saturating_sub(magnitude);
    }

    fn scroll_conversation_up(&mut self, magnitude: usize) {
        let scaled = magnitude.saturating_mul(CONVERSATION_SCROLL_SENSITIVITY);
        if self.conversation_scroll > self.conversation_max_scroll {
            self.conversation_scroll = self.conversation_max_scroll;
            return;
        }
        self.conversation_scroll = self.conversation_scroll.saturating_sub(scaled);
    }

    fn scroll_conversation_down(&mut self, magnitude: usize) {
        let scaled = magnitude.saturating_mul(CONVERSATION_SCROLL_SENSITIVITY);
        self.conversation_scroll = self.conversation_scroll.saturating_add(scaled);
    }

    fn home_index_from_mouse(&self, column: u16, row: u16) -> Option<usize> {
        let area = self.home_list_area?;
        if !contains_point(area, column, row) {
            return None;
        }
        if row <= area.y || row >= area.y + area.height - 1 {
            return None;
        }
        let local_row = (row - area.y - 1) as usize;
        let page_start = (self.home_selected_index / QUESTIONS_PER_PAGE) * QUESTIONS_PER_PAGE;
        let page_end = (page_start + QUESTIONS_PER_PAGE).min(self.total_entries());
        let count = page_end.saturating_sub(page_start);
        if local_row < count {
            Some(page_start + local_row)
        } else {
            None
        }
    }
}

fn run_app<M: LlmModelMarker, S: DatasetSplit>(
    terminal: &mut Terminal<CrosstermBackend<Stdout>>,
    mut app: App<M, S>,
) -> Result<(), Box<dyn Error>> {
    loop {
        terminal.draw(|frame| app.draw(frame))?;
        if event::poll(Duration::from_millis(33))? {
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
    }
    Ok(())
}

async fn run_model_app<M: LlmModelMarker, S: DatasetSplit>(
    config_nickname: String,
    rollout_config: RolloutConfig<S>,
    posterior_calculation_config: PosteriorCalculationConfig,
    use_tool: bool,
    fixed_temperature: NotNan<f32>,
    epoch: usize,
    override_hyperparameters: Option<PosteriorHyperparameters>,
    action_logs_path: PathBuf,
) -> Result<(), Box<dyn Error>> {
    println!("Loading keys...");
    let app = App::<M, S>::new(
        config_nickname,
        rollout_config,
        posterior_calculation_config,
        use_tool,
        fixed_temperature,
        epoch,
        override_hyperparameters,
        action_logs_path,
    )
    .await;
    println!("Keys loaded.");

    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let result = run_app::<M, S>(&mut terminal, app);

    let _ = disable_raw_mode();
    let mut stderr = io::stderr();
    let _ = execute!(
        stderr,
        LeaveAlternateScreen,
        crossterm::event::DisableMouseCapture,
        Show
    );
    result
}

fn load_action_log_config_bundle<S: DatasetSplit>(
    action_logs_path: &Path,
) -> Result<ActionLogConfigBundle<S>, io::Error> {
    let config_bundle_path = action_log_config_bundle_file_path(action_logs_path);
    if config_bundle_path.exists() {
        return read_json(&config_bundle_path).map_err(|err| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "Failed to read action log config bundle file {}: {}",
                    config_bundle_path.display(),
                    err
                ),
            )
        });
    }

    let config_paths_path = config_paths_file_path_from_action_logs_path(action_logs_path)
        .map_err(|err| io::Error::new(io::ErrorKind::InvalidInput, err))?;
    let config_paths: ConfigPaths = read_json(&config_paths_path).map_err(|err| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "Failed to read config paths file {}: {}",
                config_paths_path.display(),
                err
            ),
        )
    })?;

    let rollout_config_path = match S::dataset_file_postfix().as_str() {
        "train" => config_paths.training_rollout_config_path,
        "val" => config_paths.validation_rollout_config_path,
        "test" => config_paths.testing_rollout_config_path,
        other => {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("Unsupported dataset split postfix: {}", other),
            ));
        }
    };
    let rollout_config = read_json::<RolloutConfig<S>>(&rollout_config_path).map_err(|err| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "Failed to read rollout config file {}: {}",
                rollout_config_path, err
            ),
        )
    })?;
    let posterior_hyperparameters =
        read_json::<PosteriorHyperparameters>(crate::directories::POSTERIOR_HYPERPARAMETERS_PATH)
            .map_err(|err| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Failed to read posterior hyperparameters file: {}", err),
            )
        })?;

    Ok(ActionLogConfigBundle {
        rollout_config,
        posterior_calculation_config: PosteriorCalculationConfig {
            hyperparameters: posterior_hyperparameters,
        },
        use_tool: false,
        fixed_temperature: constants::temperature_by_split::<S>(),
    })
}

macro_rules! run_model_app_for_model_and_split {
    (
        $model_name:expr,
        $config_nickname:expr,
        $rollout_config:expr,
        $posterior_calculation_config:expr,
        $use_tool:expr,
        $fixed_temperature:expr,
        $epoch:expr,
        $override_hyperparameters:expr,
        $action_logs_path:expr,
        $split_ty:ty;
        $( $model_enum:path, $model_ty:ty ),+ $(,)?
    ) => {{
        let model_name = $model_name;
        let config_nickname = $config_nickname;
        let rollout_config = $rollout_config;
        let posterior_calculation_config = $posterior_calculation_config;
        let use_tool = $use_tool;
        let fixed_temperature = $fixed_temperature;
        let epoch = $epoch;
        let override_hyperparameters = $override_hyperparameters;
        let action_logs_path = $action_logs_path;

        let config_nickname = config_nickname.to_string();
        let rollout_config = rollout_config.clone();
        let posterior_calculation_config = posterior_calculation_config.clone();
        let override_hyperparameters = override_hyperparameters.clone();
        let action_logs_path = action_logs_path.to_path_buf();
        match model_name {
            $(
                $model_enum => {
                    run_model_app::<$model_ty, $split_ty>(
                        config_nickname,
                        rollout_config,
                        posterior_calculation_config,
                        use_tool,
                        fixed_temperature,
                        epoch,
                        override_hyperparameters,
                        action_logs_path,
                    ).await
                }
            )+
        }
    }};
}

async fn run_with_resolved_context(
    model: LlmModelName,
    dataset_split: DatasetSplitEnum,
    config_nickname: String,
    epoch: usize,
    override_hyperparameters: Option<PosteriorHyperparameters>,
    action_logs_path: PathBuf,
) -> Result<(), Box<dyn Error>> {
    match dataset_split {
        DatasetSplitEnum::Training => {
            let config_bundle = load_action_log_config_bundle::<Training>(&action_logs_path)?;
            run_model_app_for_model_and_split!(
                model,
                &config_nickname,
                config_bundle.rollout_config,
                config_bundle.posterior_calculation_config,
                config_bundle.use_tool,
                config_bundle.fixed_temperature,
                epoch,
                &override_hyperparameters,
                &action_logs_path,
                Training;
                LlmModelName::Qwen25_7b, Qwen25_7B,
                LlmModelName::Qwen3_06b, Qwen3_06B,
                LlmModelName::Qwen3_4b, Qwen3_4B,
                LlmModelName::Qwen35_4b, Qwen35_4B,
                LlmModelName::Qwen35_08b, Qwen35_08B,
                LlmModelName::Gemma3_4b, Gemma3_4BIt,
                LlmModelName::Llama31_8b, Llama31_8BInstruct,
                LlmModelName::Mistral7bInstructV03, Mistral7BInstructV03
            )
        }
        DatasetSplitEnum::Validation => {
            let config_bundle = load_action_log_config_bundle::<Validation>(&action_logs_path)?;
            run_model_app_for_model_and_split!(
                model,
                &config_nickname,
                config_bundle.rollout_config,
                config_bundle.posterior_calculation_config,
                config_bundle.use_tool,
                config_bundle.fixed_temperature,
                epoch,
                &override_hyperparameters,
                &action_logs_path,
                Validation;
                LlmModelName::Qwen25_7b, Qwen25_7B,
                LlmModelName::Qwen3_06b, Qwen3_06B,
                LlmModelName::Qwen3_4b, Qwen3_4B,
                LlmModelName::Qwen35_4b, Qwen35_4B,
                LlmModelName::Qwen35_08b, Qwen35_08B,
                LlmModelName::Gemma3_4b, Gemma3_4BIt,
                LlmModelName::Llama31_8b, Llama31_8BInstruct,
                LlmModelName::Mistral7bInstructV03, Mistral7BInstructV03
            )
        }
        DatasetSplitEnum::Testing => {
            let config_bundle = load_action_log_config_bundle::<Testing>(&action_logs_path)?;
            run_model_app_for_model_and_split!(
                model,
                &config_nickname,
                config_bundle.rollout_config,
                config_bundle.posterior_calculation_config,
                config_bundle.use_tool,
                config_bundle.fixed_temperature,
                epoch,
                &override_hyperparameters,
                &action_logs_path,
                Testing;
                LlmModelName::Qwen25_7b, Qwen25_7B,
                LlmModelName::Qwen3_06b, Qwen3_06B,
                LlmModelName::Qwen3_4b, Qwen3_4B,
                LlmModelName::Qwen35_4b, Qwen35_4B,
                LlmModelName::Qwen35_08b, Qwen35_08B,
                LlmModelName::Gemma3_4b, Gemma3_4BIt,
                LlmModelName::Llama31_8b, Llama31_8BInstruct,
                LlmModelName::Mistral7bInstructV03, Mistral7BInstructV03
            )
        }
    }
}

pub async fn run(
    action_logs_path: impl AsRef<Path>,
    override_hyperparameters: Option<PosteriorHyperparameters>,
) -> Result<(), Box<dyn Error>> {
    let action_logs_path = action_logs_path.as_ref();
    let context = parse_action_logs_context(action_logs_path)
        .map_err(|err| std::io::Error::new(std::io::ErrorKind::InvalidInput, err))?;

    run_with_resolved_context(
        context.model,
        context.dataset_split,
        context.config_nickname,
        context.epoch,
        override_hyperparameters,
        action_logs_path.to_path_buf(),
    )
    .await
}
