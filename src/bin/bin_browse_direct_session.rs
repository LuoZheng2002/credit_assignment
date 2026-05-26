use std::backtrace::Backtrace;
use std::collections::BTreeMap;
use std::error::Error;
use std::hash::{Hash, Hasher};
use std::io::{self, Stdout};

use clap::Parser;
use credit_assignment::{
    agent::{trajectory_action_types::FinalAnswer, tree::CorrectnessJudgment},
    direct_tool::{
        direct_rollout_config::DirectRolloutConfig,
        direct_tree::{ContentIndex, DirectTree, Segment, SegmentContent, SegmentId},
        direct_tree_action_log::{AssetFileDirectTreeActionLogs, DirectTreeActionLog},
        direct_tree_to_actions::TokenBranchingScore,
        posterior_calculation_config::{PosteriorCalculationConfig, PosteriorHyperparameters},
    },
    json_line_util::read_json,
    llm_model::{Gpt4o, Gpt5Mini, LlmModelName, Qwen3_4B, Qwen25, Qwen35_4B},
};
use crossterm::event::{
    self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode, KeyEvent, MouseButton,
    MouseEvent, MouseEventKind,
};
use crossterm::execute;
use crossterm::terminal::{
    EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
};
use ratatui::Terminal;
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Direction, Layout, Position, Rect};
use ratatui::prelude::Widget;
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span, Text};
use ratatui::widgets::{Block, Borders, List, ListItem, ListState, Paragraph, Wrap};
use ratatui_core::buffer::Buffer;
use research_utility::asset_file::AssetFile;
use research_utility::sqlite_store::SqliteStore;
use std::collections::hash_map::DefaultHasher;

// home page: view the questions and win rate
// the questions should be paged; each page should have 10 questions

// tree page: after clicking a question, we should enter the tree page. It should be of vertical layout.
// The top is a summary window with question, correct answer, accuracy and an optional model answer if we click on a leaf segment.
// The middle is a conversation window that shows the conversation up to the segment the user clicks on
// The bottom is the tree view like the one in src/bin/bin_browse_session.rs, but now it shows the segments instead of nodes
// The left and right arrow controls how many actions are considered to build the tree, it should demonstrate how the tree evolves with more actions applied
// We can click on a segment in the tree to show the conversation up to that segment in the conversation window;
// if a leaf segment is clicked, we can also show the model answer and the correctness judgment in the summary window.

// use the key q to transition from tree page to home page, and press q again to exit the program

#[derive(Parser, Debug)]
#[command(author, version, about = "Interactively browse rollout session logs")]
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
    override_hyperparameters_path: Option<String>,
}

const QUESTIONS_PER_PAGE: usize = 10;
const CONVERSATION_SCROLL_SENSITIVITY: usize = 4;

#[derive(Clone)]
struct QuestionEntry {
    key: usize,
    action_log: DirectTreeActionLog,
    win_rate: f64,
    num_correct: usize,
    num_leaves: usize,
}

#[derive(Clone)]
struct TreeSnapshot {
    segments: BTreeMap<SegmentId, Segment>,
    root_segment_id: SegmentId,
    leaf_segment_judgments: BTreeMap<SegmentId, CorrectnessJudgment>,
    segment_posterior_stats: BTreeMap<SegmentId, SegmentPosteriorStats>,
    segment_posterior_signal_scaled: BTreeMap<SegmentId, f32>,
    segment_posterior_mean_scaled: BTreeMap<SegmentId, f32>,
    segment_posterior_std_scaled: BTreeMap<SegmentId, f32>,
    segment_branching_score_display: BTreeMap<SegmentId, Vec<Option<f32>>>,
    segment_display_widths: BTreeMap<SegmentId, usize>,
}

#[derive(Clone, Copy)]
struct SegmentPosteriorStats {
    posterior_mean: f32,
    posterior_std: f32,
    signal_to_noise: f32,
}

impl TreeSnapshot {
    fn from_tree<M: credit_assignment::llm_model::LlmModelMarker>(
        tree: DirectTree<M>,
        width_division_ratio: Option<usize>,
        override_hyperparameters: Option<PosteriorHyperparameters>,
    ) -> Self {
        let segment_display_widths = segment_display_widths(&tree, width_division_ratio);
        let segment_posterior_stats = segment_posterior_stats(&tree, override_hyperparameters);
        let segment_posterior_signal_scaled =
            scaled_segment_posterior_signal(&tree, &segment_posterior_stats);
        let segment_posterior_mean_scaled =
            scaled_segment_posterior_mean(&tree, &segment_posterior_stats);
        let segment_posterior_std_scaled =
            scaled_segment_posterior_std(&tree, &segment_posterior_stats);
        let effective_width_division_ratio = width_division_ratio
            .unwrap_or_else(|| width_division_ratio_for_tree(&tree))
            .max(1);
        let segment_branching_score_display = segment_branching_score_display(
            &tree,
            effective_width_division_ratio,
            &segment_display_widths,
            override_hyperparameters,
        );
        let root_segment_id = tree
            .root_segment_id
            .expect("Direct tree browser requires root segment");
        Self {
            segments: tree.segments,
            root_segment_id,
            leaf_segment_judgments: tree.leaf_segment_judgments,
            segment_posterior_stats,
            segment_posterior_signal_scaled,
            segment_posterior_mean_scaled,
            segment_posterior_std_scaled,
            segment_branching_score_display,
            segment_display_widths,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mode {
    Home,
    Tree,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TreePaneFocus {
    Summary,
    Conversation,
    Tree,
}

struct TreePage {
    entry_index: usize,
    total_actions: usize,
    action_limit: usize,
    width_division_ratio: usize,
    snapshot: TreeSnapshot,
    selected_segment_id: SegmentId,
    tree_lines: Vec<String>,
    rendered_segments: Vec<TreeRenderedSegment>,
    hovered_segment_id: Option<SegmentId>,
    tree_area: Option<Rect>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TreeScrollMode {
    Scaling,
    Panning,
    Evolution,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TreeColorMode {
    SignalToNoise,
    BranchingScore,
    PosteriorMean,
    PosteriorStd,
}

impl TreeColorMode {
    fn label(self) -> &'static str {
        match self {
            Self::SignalToNoise => "SignalToNoise",
            Self::BranchingScore => "BranchingScore",
            Self::PosteriorMean => "PosteriorMean",
            Self::PosteriorStd => "PosteriorStd",
        }
    }
}

impl TreeScrollMode {
    fn label(self) -> &'static str {
        match self {
            Self::Scaling => "Scaling",
            Self::Panning => "Panning",
            Self::Evolution => "Evolution",
        }
    }
}

#[derive(Clone, Copy)]
struct TreeRenderedSegment {
    segment_id: SegmentId,
    row: usize,
    col: usize,
    width: usize,
}

struct ConversationRender {
    plain: String,
    styled: Text<'static>,
}

impl TreePage {
    fn new(
        model: LlmModelName,
        entry_index: usize,
        entry: &QuestionEntry,
        override_hyperparameters: Option<PosteriorHyperparameters>,
    ) -> Self {
        let total_actions = entry.action_log.actions.len();
        let action_limit = total_actions;
        let width_division_ratio = width_division_ratio_for_model(model, &entry.action_log);
        let snapshot = snapshot_for_model(
            model,
            &entry.action_log,
            action_limit,
            Some(width_division_ratio),
            override_hyperparameters,
        );
        let selected_segment_id = snapshot.root_segment_id;
        let (tree_lines, rendered_segments) = build_segment_graph_lines(&snapshot);
        Self {
            entry_index,
            total_actions,
            action_limit,
            width_division_ratio,
            snapshot,
            selected_segment_id,
            tree_lines,
            rendered_segments,
            hovered_segment_id: None,
            tree_area: None,
        }
    }

    fn rebuild_snapshot(
        &mut self,
        model: LlmModelName,
        entry: &QuestionEntry,
        override_hyperparameters: Option<PosteriorHyperparameters>,
    ) {
        self.snapshot = snapshot_for_model(
            model,
            &entry.action_log,
            self.action_limit,
            Some(self.width_division_ratio),
            override_hyperparameters,
        );
        let (tree_lines, rendered_segments) = build_segment_graph_lines(&self.snapshot);
        self.tree_lines = tree_lines;
        self.rendered_segments = rendered_segments;
        if !self
            .snapshot
            .segments
            .contains_key(&self.selected_segment_id)
        {
            self.selected_segment_id = self.snapshot.root_segment_id;
        }
        if self
            .hovered_segment_id
            .is_some_and(|id| !self.snapshot.segments.contains_key(&id))
        {
            self.hovered_segment_id = None;
        }
    }

    fn set_action_limit(
        &mut self,
        model: LlmModelName,
        entry: &QuestionEntry,
        new_limit: usize,
        override_hyperparameters: Option<PosteriorHyperparameters>,
    ) {
        if self.action_limit == new_limit {
            return;
        }
        self.action_limit = new_limit;
        self.rebuild_snapshot(model, entry, override_hyperparameters);
    }

    fn set_width_division_ratio(
        &mut self,
        model: LlmModelName,
        entry: &QuestionEntry,
        new_ratio: usize,
        override_hyperparameters: Option<PosteriorHyperparameters>,
    ) {
        if self.width_division_ratio == new_ratio {
            return;
        }
        self.width_division_ratio = new_ratio.max(1);
        self.rebuild_snapshot(model, entry, override_hyperparameters);
    }
}

struct App {
    model: LlmModelName,
    override_hyperparameters: Option<PosteriorHyperparameters>,
    action_log_store: SqliteStore<usize, DirectTreeActionLog>,
    entry_keys: Vec<usize>,
    entry_cache: Vec<Option<QuestionEntry>>,
    mode: Mode,
    home_selected_index: usize,
    home_list_area: Option<Rect>,
    summary_area: Option<Rect>,
    conversation_area: Option<Rect>,
    tree_page: Option<TreePage>,
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
}

impl App {
    fn new(
        model: LlmModelName,
        action_log_store: SqliteStore<usize, DirectTreeActionLog>,
        entry_keys: Vec<usize>,
        override_hyperparameters: Option<PosteriorHyperparameters>,
    ) -> Self {
        let entry_cache = vec![None; entry_keys.len()];
        Self {
            model,
            override_hyperparameters,
            action_log_store,
            entry_keys,
            entry_cache,
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
        }
    }

    fn total_entries(&self) -> usize {
        self.entry_keys.len()
    }

    fn ensure_entry_loaded(&mut self, index: usize) {
        if index >= self.total_entries() || self.entry_cache[index].is_some() {
            return;
        }
        let key = self.entry_keys[index];
        let action_log = tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(async {
                self.action_log_store
                    .get(key)
                    .await
                    .unwrap()
                    .expect("key from sqlite key set must exist")
            })
        });
        let (num_correct, num_leaves, win_rate) = question_stats_from_action_log(self.model, &action_log);
        self.entry_cache[index] = Some(QuestionEntry {
            key,
            action_log,
            win_rate,
            num_correct,
            num_leaves,
        });
    }

    fn ensure_visible_home_page_loaded(&mut self) {
        if self.total_entries() == 0 {
            return;
        }
        let page_start = (self.home_selected_index / QUESTIONS_PER_PAGE) * QUESTIONS_PER_PAGE;
        let page_end = (page_start + QUESTIONS_PER_PAGE).min(self.total_entries());
        for index in page_start..page_end {
            self.ensure_entry_loaded(index);
        }
    }

    fn loaded_entry(&self, index: usize) -> &QuestionEntry {
        self.entry_cache[index]
            .as_ref()
            .expect("entry must be loaded before use")
    }

    fn draw(&mut self, frame: &mut ratatui::Frame<'_>) {
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
            self.ensure_visible_home_page_loaded();
            let page_start = (self.home_selected_index / QUESTIONS_PER_PAGE) * QUESTIONS_PER_PAGE;
            let page_end = (page_start + QUESTIONS_PER_PAGE).min(self.total_entries());
            let page_entries: Vec<&QuestionEntry> = (page_start..page_end)
                .map(|index| self.loaded_entry(index))
                .collect();

            let items: Vec<ListItem> = page_entries
                .iter()
                .map(|entry| {
                    let question_preview =
                        single_line_preview(&entry.action_log.question.question, 72);
                    ListItem::new(format!(
                        "#{}  win {:>5.1}% ({}/{})  {}",
                        entry.key,
                        entry.win_rate * 100.0,
                        entry.num_correct,
                        entry.num_leaves,
                        question_preview
                    ))
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
        self.ensure_entry_loaded(entry_index);
        let entry = self.loaded_entry(entry_index).clone();

        let Some(tree_page) = self.tree_page.as_mut() else {
            return;
        };
        let selected_segment = tree_page
            .snapshot
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
            .snapshot
            .leaf_segment_judgments
            .get(&tree_page.selected_segment_id);
        let posterior_stats = tree_page
            .snapshot
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

        let mut summary = format!(
            "Question #{}\nQuestion: {}\nCorrect answer: {}\nActions applied: {}/{}\nSelected segment: S{} (children: {})\nposterior_mean: {}\nposterior_std: {}\nsignal_to_noise: {}",
            entry.key,
            entry.action_log.question.question,
            entry.action_log.question.correct_answer,
            tree_page.action_limit,
            tree_page.total_actions,
            tree_page.selected_segment_id.0,
            selected_segment.child_ids.len(),
            posterior_mean_text,
            posterior_std_text,
            signal_to_noise_text,
        );
        match self.tree_color_mode {
            TreeColorMode::SignalToNoise => {
                if let Some((min_value, max_value)) =
                    posterior_stat_min_max(&tree_page.snapshot.segment_posterior_stats, |stats| {
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
                    posterior_stat_min_max(&tree_page.snapshot.segment_posterior_stats, |stats| {
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
                    posterior_stat_min_max(&tree_page.snapshot.segment_posterior_stats, |stats| {
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
        }
        if let Some(judgment) = judgment {
            let model_answer = model_answer_text(&judgment.model_answer);
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

        let conversation_render =
            build_conversation_render(&tree_page.snapshot, tree_page.selected_segment_id);
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
                    "Segment tree [scroll:{} color:{} ratio:{} hscroll:{} actions:{}/{}] (1:scale 2:pan 3:evolve, 4:snr 5:branch 6:mean 7:std, wheel follows scroll mode)",
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
                    self.home_selected_index =
                        (self.home_selected_index + QUESTIONS_PER_PAGE)
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
                } else if let Some(tree_area) = self.tree_page.as_ref().and_then(|page| page.tree_area)
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
            tree_segment_at_mouse(tree_page, mouse.column, mouse.row, self.tree_horizontal_scroll)
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
            MouseEventKind::ScrollUp => match self.tree_focus {
                TreePaneFocus::Summary => self.scroll_summary_up(1),
                TreePaneFocus::Conversation => self.scroll_conversation_up(1),
                TreePaneFocus::Tree => {
                    let entry_index = self
                        .tree_page
                        .as_ref()
                        .expect("tree page must exist in tree mode")
                        .entry_index;
                    self.ensure_entry_loaded(entry_index);
                    let entry = self.loaded_entry(entry_index).clone();
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
                                    self.model,
                                    &entry,
                                    new_ratio,
                                    self.override_hyperparameters,
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
                                self.model,
                                &entry,
                                next,
                                self.override_hyperparameters,
                            );
                        }
                    }
                }
            },
            MouseEventKind::ScrollDown => match self.tree_focus {
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
                    self.ensure_entry_loaded(entry_index);
                    let entry = self.loaded_entry(entry_index).clone();
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
                                self.model,
                                &entry,
                                new_ratio,
                                self.override_hyperparameters,
                            );
                        }
                        TreeScrollMode::Panning => {
                            self.tree_horizontal_scroll =
                                self.tree_horizontal_scroll.saturating_add(4);
                        }
                        TreeScrollMode::Evolution => {
                            let next = (tree_page.action_limit + 1).min(tree_page.total_actions);
                            tree_page.set_action_limit(
                                self.model,
                                &entry,
                                next,
                                self.override_hyperparameters,
                            );
                        }
                    }
                }
            },
            _ => {}
        }
    }

    fn open_selected_home_entry(&mut self) {
        if self.total_entries() == 0 {
            return;
        }
        self.ensure_entry_loaded(self.home_selected_index);
        let entry = self.loaded_entry(self.home_selected_index).clone();
        self.tree_page = Some(TreePage::new(
            self.model,
            self.home_selected_index,
            &entry,
            self.override_hyperparameters,
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

fn contains_point(area: Rect, x: u16, y: u16) -> bool {
    x >= area.x && x < area.x + area.width && y >= area.y && y < area.y + area.height
}

fn clamp_scroll(value: usize) -> u16 {
    if value > u16::MAX as usize {
        u16::MAX
    } else {
        value as u16
    }
}

fn scale_horizontal_scroll(scroll: usize, old_ratio: usize, new_ratio: usize) -> usize {
    let safe_old = old_ratio.max(1) as f64;
    let safe_new = new_ratio.max(1) as f64;
    ((scroll as f64) * safe_old / safe_new).round() as usize
}

fn segment_token_length(segment: &Segment) -> usize {
    segment
        .content
        .iter()
        .map(|content| match content {
            SegmentContent::Prompt(tokens) => tokens.tokens.len(),
            SegmentContent::ReasoningOrToolCall { tokens, .. } => tokens.tokens.len(),
            SegmentContent::ToolResponse(tokens) => tokens.tokens.len(),
        })
        .sum()
}

fn trajectory_token_length_to_root<M: credit_assignment::llm_model::LlmModelMarker>(
    tree: &DirectTree<M>,
    leaf_segment_id: SegmentId,
) -> Option<usize> {
    let mut total = 0usize;
    let mut cursor = Some(leaf_segment_id);
    while let Some(segment_id) = cursor {
        let segment = tree.segments.get(&segment_id)?;
        total = total.saturating_add(segment_token_length(segment));
        cursor = segment.parent_id;
    }
    Some(total)
}

fn fixed_width_scale_ratio_for_tree<M: credit_assignment::llm_model::LlmModelMarker>(
    tree: &DirectTree<M>,
) -> f64 {
    let mut trunk_trajectory_lengths: Vec<usize> = tree
        .trunk_leaf_segments
        .iter()
        .filter_map(|leaf_id| trajectory_token_length_to_root(tree, *leaf_id))
        .collect();
    if trunk_trajectory_lengths.is_empty() {
        trunk_trajectory_lengths = tree
            .leaf_segment_judgments
            .keys()
            .copied()
            .filter_map(|leaf_id| trajectory_token_length_to_root(tree, leaf_id))
            .collect();
    }

    let avg_trunk_trajectory_len = if trunk_trajectory_lengths.is_empty() {
        1.0
    } else {
        trunk_trajectory_lengths
            .iter()
            .map(|value| *value as f64)
            .sum::<f64>()
            / trunk_trajectory_lengths.len() as f64
    };
    let safe_avg = if avg_trunk_trajectory_len <= 0.0 {
        1.0
    } else {
        avg_trunk_trajectory_len
    };
    100.0_f64 / safe_avg
}

fn segment_display_widths<M: credit_assignment::llm_model::LlmModelMarker>(
    tree: &DirectTree<M>,
    width_division_ratio: Option<usize>,
) -> BTreeMap<SegmentId, usize> {
    let mut widths = BTreeMap::new();
    if tree.segments.is_empty() {
        return widths;
    }

    let division_ratio =
        width_division_ratio.unwrap_or_else(|| width_division_ratio_for_tree(tree));

    for (segment_id, segment) in &tree.segments {
        let token_len = segment_token_length(segment);
        let scaled = token_len / division_ratio.max(1);
        widths.insert(*segment_id, scaled.max(1));
    }
    widths
}

fn segment_branching_score_display<M: credit_assignment::llm_model::LlmModelMarker>(
    tree: &DirectTree<M>,
    width_division_ratio: usize,
    segment_display_widths: &BTreeMap<SegmentId, usize>,
    override_hyperparameters: Option<PosteriorHyperparameters>,
) -> BTreeMap<SegmentId, Vec<Option<f32>>> {
    let mut displays = BTreeMap::new();
    if tree.segments.is_empty() {
        return displays;
    }

    if tree.leaf_segment_judgments.is_empty() {
        for (segment_id, width) in segment_display_widths {
            displays.insert(*segment_id, vec![None; (*width).max(1)]);
        }
        return displays;
    }

    let posteriors = tree.calculate_segment_posteriors(override_hyperparameters);
    let mut segment_uncertainty_scores = tree.posteriors_to_segment_uncertainty_scores(&posteriors);
    for segment_id in tree.segments.keys().copied() {
        segment_uncertainty_scores.entry(segment_id).or_insert(0.0);
    }
    let average_trunk_token_length = if tree.trunk_token_lengths.is_empty() {
        1.0
    } else {
        tree.trunk_token_lengths.values().sum::<usize>() as f32
            / tree.trunk_token_lengths.len() as f32
    };
    let per_token_branching_scores: BTreeMap<
        SegmentId,
        BTreeMap<ContentIndex, BTreeMap<usize, TokenBranchingScore>>,
    > = tree.calculate_per_token_branching_scores(
        &segment_uncertainty_scores,
        average_trunk_token_length,
    );

    for (segment_id, segment) in &tree.segments {
        let mut token_level_scores: Vec<Option<f32>> = Vec::new();
        for (content_index, content) in segment.content.iter().enumerate() {
            match content {
                SegmentContent::Prompt(tokens) | SegmentContent::ToolResponse(tokens) => {
                    token_level_scores.extend((0..tokens.tokens.len()).map(|_| None));
                }
                SegmentContent::ReasoningOrToolCall { tokens, .. } => {
                    for token_offset in 0..tokens.tokens.len() {
                        let score = per_token_branching_scores
                            .get(segment_id)
                            .and_then(|content_map| content_map.get(&content_index))
                            .and_then(|offset_map| offset_map.get(&token_offset))
                            .map(|entry| entry.branching_score)
                            .unwrap_or(0.0)
                            .clamp(0.0, 1.0);
                        token_level_scores.push(Some(score));
                    }
                }
            }
        }

        let display_width = segment_display_widths
            .get(segment_id)
            .copied()
            .unwrap_or(1)
            .max(1);
        let ratio = width_division_ratio.max(1);
        let mut display_scores: Vec<Option<f32>> = Vec::with_capacity(display_width);
        for display_index in 0..display_width {
            let start = display_index.saturating_mul(ratio);
            if start >= token_level_scores.len() {
                display_scores.push(None);
                continue;
            }
            let end = (start + ratio).min(token_level_scores.len());
            let max_value = token_level_scores[start..end]
                .iter()
                .flatten()
                .copied()
                .max_by(|a, b| a.partial_cmp(b).unwrap());
            display_scores.push(max_value.map(|value| value.clamp(0.0, 1.0)));
        }
        displays.insert(*segment_id, display_scores);
    }
    displays
}

fn segment_posterior_stats<M: credit_assignment::llm_model::LlmModelMarker>(
    tree: &DirectTree<M>,
    override_hyperparameters: Option<PosteriorHyperparameters>,
) -> BTreeMap<SegmentId, SegmentPosteriorStats> {
    let mut stats_by_segment = BTreeMap::new();
    if tree.segments.is_empty() || tree.leaf_segment_judgments.is_empty() {
        return stats_by_segment;
    }

    let eps = 1e-8_f32;
    let posteriors = tree.calculate_segment_posteriors(override_hyperparameters);
    for (segment_id, posterior) in posteriors {
        let posterior_std = posterior.log_std.exp();
        let signal_to_noise = posterior.mean / (posterior_std + eps);
        stats_by_segment.insert(
            segment_id,
            SegmentPosteriorStats {
                posterior_mean: posterior.mean,
                posterior_std,
                signal_to_noise,
            },
        );
    }
    stats_by_segment
}

fn scaled_segment_posterior_signal<M: credit_assignment::llm_model::LlmModelMarker>(
    tree: &DirectTree<M>,
    stats_by_segment: &BTreeMap<SegmentId, SegmentPosteriorStats>,
) -> BTreeMap<SegmentId, f32> {
    scaled_segment_posterior_value(tree, stats_by_segment, |stats| stats.signal_to_noise)
}

fn scaled_segment_posterior_mean<M: credit_assignment::llm_model::LlmModelMarker>(
    tree: &DirectTree<M>,
    stats_by_segment: &BTreeMap<SegmentId, SegmentPosteriorStats>,
) -> BTreeMap<SegmentId, f32> {
    scaled_segment_posterior_value(tree, stats_by_segment, |stats| stats.posterior_mean)
}

fn scaled_segment_posterior_std<M: credit_assignment::llm_model::LlmModelMarker>(
    tree: &DirectTree<M>,
    stats_by_segment: &BTreeMap<SegmentId, SegmentPosteriorStats>,
) -> BTreeMap<SegmentId, f32> {
    scaled_segment_posterior_value(tree, stats_by_segment, |stats| stats.posterior_std)
}

fn scaled_segment_posterior_value<M: credit_assignment::llm_model::LlmModelMarker>(
    tree: &DirectTree<M>,
    stats_by_segment: &BTreeMap<SegmentId, SegmentPosteriorStats>,
    value_selector: impl Fn(SegmentPosteriorStats) -> f32,
) -> BTreeMap<SegmentId, f32> {
    let mut raw_values = BTreeMap::new();
    for segment_id in tree.segments.keys().copied() {
        raw_values.insert(segment_id, 0.0);
    }
    if tree.segments.is_empty() || stats_by_segment.is_empty() {
        return raw_values;
    }

    for (segment_id, stats) in stats_by_segment {
        raw_values.insert(*segment_id, value_selector(*stats));
    }

    let mut max_abs = 0.0_f32;
    for value in raw_values.values().copied() {
        max_abs = max_abs.max(value.abs());
    }
    if max_abs <= 0.0 {
        return raw_values;
    }

    raw_values
        .into_iter()
        .map(|(segment_id, value)| (segment_id, value / max_abs))
        .collect()
}

fn signed_posterior_to_color(signed_position: f32) -> Color {
    let x = signed_position.clamp(-1.0, 1.0);
    let (red, green) = if x >= 0.0 {
        (((1.0 - x) * 255.0).round() as u8, 255)
    } else {
        (255, ((1.0 + x) * 255.0).round() as u8)
    };
    Color::Rgb(red, green, 0)
}

fn posterior_stat_min_max(
    stats_by_segment: &BTreeMap<SegmentId, SegmentPosteriorStats>,
    selector: impl Fn(SegmentPosteriorStats) -> f32,
) -> Option<(f32, f32)> {
    let mut min_value = f32::INFINITY;
    let mut max_value = f32::NEG_INFINITY;
    for stats in stats_by_segment.values().copied() {
        let value = selector(stats);
        min_value = min_value.min(value);
        max_value = max_value.max(value);
    }
    if min_value.is_finite() && max_value.is_finite() {
        Some((min_value, max_value))
    } else {
        None
    }
}

fn branching_score_to_color(score: f32) -> Color {
    let x = score.clamp(0.0, 1.0);
    if x <= 0.5 {
        let green = ((x / 0.5) * 255.0).round() as u8;
        Color::Rgb(255, green, 0)
    } else {
        let red = (((1.0 - x) / 0.5) * 255.0).round() as u8;
        Color::Rgb(red, 255, 0)
    }
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

#[derive(Clone, Copy)]
struct PaneMetrics {
    text_hash: u64,
    width: u16,
    height: u16,
    lines: usize,
}

fn compute_wrapped_line_count(
    text: &str,
    area: Rect,
    metrics_slot: &mut Option<PaneMetrics>,
) -> usize {
    if area.width == 0 {
        return 0;
    }
    let mut hasher = DefaultHasher::new();
    text.hash(&mut hasher);
    let text_hash = hasher.finish();
    if let Some(metrics) = metrics_slot {
        if metrics.text_hash == text_hash
            && metrics.width == area.width
            && metrics.height == area.height
        {
            return metrics.lines;
        }
    }
    let lines = count_wrapped_lines(text, area);
    *metrics_slot = Some(PaneMetrics {
        text_hash,
        width: area.width,
        height: area.height,
        lines,
    });
    lines
}

fn single_line_preview(text: &str, max_chars: usize) -> String {
    let compact = text.split_whitespace().collect::<Vec<_>>().join(" ");
    let chars: Vec<char> = compact.chars().collect();
    if chars.len() <= max_chars {
        compact
    } else {
        let mut prefix: String = chars
            .into_iter()
            .take(max_chars.saturating_sub(3))
            .collect();
        prefix.push_str("...");
        prefix
    }
}

fn model_answer_text(answer: &FinalAnswer) -> &str {
    match answer {
        FinalAnswer::ModelProvided(text) => text,
        FinalAnswer::Failure(text) => text,
    }
}

fn write_pattern(canvas: &mut [Vec<char>], row: usize, col: usize, pattern: &str) {
    assert!(row < canvas.len(), "row out of bounds while drawing tree");
    let pattern_chars: Vec<char> = pattern.chars().collect();
    let required_width = col + pattern_chars.len();
    if canvas[row].len() < required_width {
        canvas[row].resize(required_width, ' ');
    }
    for (i, ch) in pattern_chars.into_iter().enumerate() {
        canvas[row][col + i] = ch;
    }
}

fn collect_leaf_order(snapshot: &TreeSnapshot) -> Vec<SegmentId> {
    let mut leaves = Vec::new();
    let mut stack = vec![snapshot.root_segment_id];
    while let Some(segment_id) = stack.pop() {
        let segment = snapshot
            .segments
            .get(&segment_id)
            .expect("segment in snapshot must exist");
        if segment.child_ids.is_empty() {
            leaves.push(segment_id);
        } else {
            for child in segment.child_ids.iter().rev() {
                stack.push(*child);
            }
        }
    }
    leaves
}

fn path_root_to_segment(snapshot: &TreeSnapshot, segment_id: SegmentId) -> Vec<SegmentId> {
    let mut path = Vec::new();
    let mut cursor = Some(segment_id);
    while let Some(current_id) = cursor {
        path.push(current_id);
        cursor = snapshot
            .segments
            .get(&current_id)
            .expect("segment in path must exist")
            .parent_id;
    }
    path.reverse();
    path
}

fn build_segment_graph_lines(snapshot: &TreeSnapshot) -> (Vec<String>, Vec<TreeRenderedSegment>) {
    let ordered_leaf_ids = collect_leaf_order(snapshot);
    let line_count = ordered_leaf_ids.len().max(1);
    let mut canvas: Vec<Vec<char>> = vec![Vec::new(); line_count];

    let mut row_by_segment: BTreeMap<SegmentId, usize> = BTreeMap::new();
    for (row, leaf_id) in ordered_leaf_ids.iter().copied().enumerate() {
        for segment_id in path_root_to_segment(snapshot, leaf_id) {
            row_by_segment.entry(segment_id).or_insert(row);
        }
    }

    if row_by_segment.is_empty() {
        row_by_segment.insert(snapshot.root_segment_id, 0);
    }

    let mut width_by_segment: BTreeMap<SegmentId, usize> = BTreeMap::new();
    for segment_id in snapshot.segments.keys().copied() {
        let width = snapshot
            .segment_display_widths
            .get(&segment_id)
            .copied()
            .unwrap_or(1)
            .max(1);
        width_by_segment.insert(segment_id, width);
    }

    let mut col_by_segment: BTreeMap<SegmentId, usize> = BTreeMap::new();
    col_by_segment.insert(snapshot.root_segment_id, 0);
    let mut stack = vec![snapshot.root_segment_id];
    while let Some(parent_id) = stack.pop() {
        let parent_col = *col_by_segment
            .get(&parent_id)
            .expect("parent column must exist");
        let parent_width = *width_by_segment
            .get(&parent_id)
            .expect("parent width must exist");
        let child_col = parent_col + parent_width + 3;
        let parent = snapshot
            .segments
            .get(&parent_id)
            .expect("parent segment must exist in snapshot");
        for child_id in parent.child_ids.iter().rev() {
            if col_by_segment.contains_key(child_id) {
                continue;
            }
            col_by_segment.insert(*child_id, child_col);
            stack.push(*child_id);
        }
    }
    for segment_id in snapshot.segments.keys().copied() {
        if col_by_segment.contains_key(&segment_id) {
            continue;
        }
        let path = path_root_to_segment(snapshot, segment_id);
        let mut col = 0usize;
        for parent_id in path.iter().take(path.len().saturating_sub(1)) {
            col = col
                .saturating_add(width_by_segment.get(parent_id).copied().unwrap_or(1))
                .saturating_add(3);
        }
        col_by_segment.insert(segment_id, col);
    }

    let mut rendered_segments = Vec::new();
    for segment_id in snapshot.segments.keys().copied() {
        let Some(&row) = row_by_segment.get(&segment_id) else {
            continue;
        };
        let col = *col_by_segment
            .get(&segment_id)
            .expect("segment col must be available");
        let width = *width_by_segment
            .get(&segment_id)
            .expect("segment width must be available");
        write_pattern(&mut canvas, row, col, &"=".repeat(width));
        rendered_segments.push(TreeRenderedSegment {
            segment_id,
            row,
            col,
            width,
        });
    }

    for (parent_id, parent) in &snapshot.segments {
        let Some(&parent_row) = row_by_segment.get(parent_id) else {
            continue;
        };
        let parent_col = *col_by_segment
            .get(parent_id)
            .expect("parent col must be available");
        let parent_width = *width_by_segment
            .get(parent_id)
            .expect("parent width must be available");
        let edge_col = parent_col + parent_width;

        for child_id in parent.child_ids.iter().copied() {
            let child_row = *row_by_segment
                .get(&child_id)
                .expect("child row must be available");
            let child_col = *col_by_segment
                .get(&child_id)
                .expect("child col must be available");
            assert!(
                child_col > edge_col,
                "child column must be right of parent edge"
            );
            let junction_col = edge_col + 1;
            if child_row == parent_row {
                let has_lower_sibling = parent
                    .child_ids
                    .iter()
                    .copied()
                    .filter(|id| *id != child_id)
                    .any(|other_id| {
                        row_by_segment
                            .get(&other_id)
                            .is_some_and(|other_row| *other_row > parent_row)
                    });
                for col in edge_col..child_col {
                    if canvas[parent_row].len() <= col {
                        canvas[parent_row].resize(col + 1, ' ');
                    }
                    if canvas[parent_row][col] == ' ' {
                        canvas[parent_row][col] = '━';
                    }
                }
                if has_lower_sibling {
                    if canvas[parent_row].len() <= junction_col {
                        canvas[parent_row].resize(junction_col + 1, ' ');
                    }
                    canvas[parent_row][junction_col] = '┳';
                }
            } else {
                for row in (parent_row + 1)..child_row {
                    if canvas[row].len() <= junction_col {
                        canvas[row].resize(junction_col + 1, ' ');
                    }
                    if canvas[row][junction_col] == ' ' {
                        canvas[row][junction_col] = '┃';
                    }
                }
                let has_lower_sibling = parent
                    .child_ids
                    .iter()
                    .copied()
                    .filter(|id| *id != child_id)
                    .any(|other_id| {
                        row_by_segment
                            .get(&other_id)
                            .is_some_and(|other_row| *other_row > child_row)
                    });
                let connector = if has_lower_sibling { '┣' } else { '┗' };
                if canvas[child_row].len() <= junction_col {
                    canvas[child_row].resize(junction_col + 1, ' ');
                }
                canvas[child_row][junction_col] = connector;
                for col in (junction_col + 1)..child_col {
                    if canvas[child_row].len() <= col {
                        canvas[child_row].resize(col + 1, ' ');
                    }
                    if canvas[child_row][col] == ' ' {
                        canvas[child_row][col] = '━';
                    }
                }
            }
        }
    }

    for leaf_id in ordered_leaf_ids {
        let row = *row_by_segment
            .get(&leaf_id)
            .expect("leaf row must be available");
        let col = *col_by_segment
            .get(&leaf_id)
            .expect("leaf col must be available")
            + width_by_segment
                .get(&leaf_id)
                .copied()
                .expect("leaf width must be available");
        let suffix = if let Some(judgment) = snapshot.leaf_segment_judgments.get(&leaf_id) {
            if judgment.is_correct {
                "━━━✓"
            } else {
                "━━━✗"
            }
        } else {
            "━━━?"
        };
        write_pattern(&mut canvas, row, col, suffix);
    }

    let mut lines = Vec::with_capacity(canvas.len());
    for row in canvas {
        let mut line: String = row.into_iter().collect();
        line = line.trim_end().to_string();
        lines.push(line);
    }

    (lines, rendered_segments)
}

fn render_tree_line(
    tree_page: &TreePage,
    row: usize,
    horizontal_scroll: usize,
    color_mode: TreeColorMode,
) -> Line<'static> {
    let line = tree_page
        .tree_lines
        .get(row)
        .expect("tree row must be in bounds");
    let line_chars: Vec<char> = line.chars().collect();
    if horizontal_scroll >= line_chars.len() {
        return Line::from(String::new());
    }
    let mut styles: Vec<Option<Style>> = vec![None; line_chars.len()];

    for rendered in tree_page.rendered_segments.iter().copied() {
        if rendered.row != row {
            continue;
        }
        let is_selected = rendered.segment_id == tree_page.selected_segment_id;
        let is_hovered = tree_page.hovered_segment_id == Some(rendered.segment_id);
        match color_mode {
            TreeColorMode::SignalToNoise => {
                let posterior_signal = tree_page
                    .snapshot
                    .segment_posterior_signal_scaled
                    .get(&rendered.segment_id)
                    .copied()
                    .unwrap_or(0.0);
                let mut style = Style::default().fg(signed_posterior_to_color(posterior_signal));
                if is_selected {
                    style = style.add_modifier(Modifier::BOLD | Modifier::UNDERLINED);
                }
                if is_hovered {
                    style = style.bg(Color::DarkGray);
                }
                for idx in rendered.col..(rendered.col + rendered.width) {
                    if idx < styles.len() {
                        styles[idx] = Some(style);
                    }
                }
            }
            TreeColorMode::PosteriorMean => {
                let posterior_mean = tree_page
                    .snapshot
                    .segment_posterior_mean_scaled
                    .get(&rendered.segment_id)
                    .copied()
                    .unwrap_or(0.0);
                let mut style = Style::default().fg(signed_posterior_to_color(posterior_mean));
                if is_selected {
                    style = style.add_modifier(Modifier::BOLD | Modifier::UNDERLINED);
                }
                if is_hovered {
                    style = style.bg(Color::DarkGray);
                }
                for idx in rendered.col..(rendered.col + rendered.width) {
                    if idx < styles.len() {
                        styles[idx] = Some(style);
                    }
                }
            }
            TreeColorMode::PosteriorStd => {
                let posterior_std = tree_page
                    .snapshot
                    .segment_posterior_std_scaled
                    .get(&rendered.segment_id)
                    .copied()
                    .unwrap_or(0.0);
                let mut style = Style::default().fg(signed_posterior_to_color(posterior_std));
                if is_selected {
                    style = style.add_modifier(Modifier::BOLD | Modifier::UNDERLINED);
                }
                if is_hovered {
                    style = style.bg(Color::DarkGray);
                }
                for idx in rendered.col..(rendered.col + rendered.width) {
                    if idx < styles.len() {
                        styles[idx] = Some(style);
                    }
                }
            }
            TreeColorMode::BranchingScore => {
                let score_by_display_token = tree_page
                    .snapshot
                    .segment_branching_score_display
                    .get(&rendered.segment_id);
                for display_offset in 0..rendered.width {
                    let idx = rendered.col + display_offset;
                    if idx >= styles.len() {
                        continue;
                    }
                    let score = score_by_display_token
                        .and_then(|scores| scores.get(display_offset))
                        .and_then(|entry| *entry);
                    let mut style = match score {
                        Some(score) => Style::default().fg(branching_score_to_color(score)),
                        None => Style::default().fg(Color::White),
                    };
                    if is_selected {
                        style = style.add_modifier(Modifier::BOLD | Modifier::UNDERLINED);
                    }
                    if is_hovered {
                        style = style.bg(Color::DarkGray);
                    }
                    styles[idx] = Some(style);
                }
            }
        }
    }

    for (idx, ch) in line_chars.iter().copied().enumerate() {
        if ch == '✓' {
            styles[idx] = Some(
                Style::default()
                    .fg(Color::Green)
                    .add_modifier(Modifier::BOLD),
            );
        }
        if ch == '✗' {
            styles[idx] = Some(Style::default().fg(Color::Red).add_modifier(Modifier::BOLD));
        }
    }

    let mut spans = Vec::new();
    let mut segment_text = String::new();
    let mut segment_style = styles[horizontal_scroll];
    for idx in horizontal_scroll..line_chars.len() {
        let ch = line_chars[idx];
        let style = styles[idx];
        if style != segment_style {
            if !segment_text.is_empty() {
                spans.push(match segment_style {
                    Some(style) => Span::styled(segment_text.clone(), style),
                    None => Span::raw(segment_text.clone()),
                });
                segment_text.clear();
            }
            segment_style = style;
        }
        segment_text.push(ch);
    }
    if !segment_text.is_empty() {
        spans.push(match segment_style {
            Some(style) => Span::styled(segment_text, style),
            None => Span::raw(segment_text),
        });
    }
    Line::from(spans)
}

fn tree_segment_at_mouse(
    tree_page: &TreePage,
    column: u16,
    row: u16,
    horizontal_scroll: usize,
) -> Option<SegmentId> {
    let area = tree_page.tree_area?;
    if !contains_point(area, column, row) {
        return None;
    }
    if row <= area.y || row >= area.y + area.height - 1 {
        return None;
    }
    if column <= area.x || column >= area.x + area.width - 1 {
        return None;
    }

    let local_row = (row - area.y - 1) as usize;
    let local_col = (column - area.x - 1) as usize + horizontal_scroll;
    tree_page
        .rendered_segments
        .iter()
        .copied()
        .find(|rendered| {
            rendered.row == local_row
                && local_col >= rendered.col
                && local_col < rendered.col + rendered.width
        })
        .map(|rendered| rendered.segment_id)
}

fn push_text_as_spans(
    lines: &mut Vec<Vec<Span<'static>>>,
    text: &str,
    style: Option<Style>,
    plain: &mut String,
) {
    if text.is_empty() {
        return;
    }
    plain.push_str(text);
    if lines.is_empty() {
        lines.push(Vec::new());
    }
    for (index, part) in text.split('\n').enumerate() {
        if index > 0 {
            lines.push(Vec::new());
        }
        if part.is_empty() {
            continue;
        }
        let span = match style {
            Some(style) => Span::styled(part.to_string(), style),
            None => Span::raw(part.to_string()),
        };
        let current_line = lines
            .last_mut()
            .expect("lines must contain at least one entry while appending text");
        current_line.push(span);
    }
}

fn build_conversation_render(snapshot: &TreeSnapshot, segment_id: SegmentId) -> ConversationRender {
    let mut path = Vec::new();
    let mut cursor = Some(segment_id);
    while let Some(current_id) = cursor {
        path.push(current_id);
        cursor = snapshot
            .segments
            .get(&current_id)
            .expect("path segment must exist")
            .parent_id;
    }
    path.reverse();

    let mut plain = String::new();
    let mut lines: Vec<Vec<Span<'static>>> = vec![Vec::new()];
    for sid in path {
        let segment = snapshot
            .segments
            .get(&sid)
            .expect("segment in path must exist");
        let segment_color = snapshot
            .segment_posterior_signal_scaled
            .get(&sid)
            .copied()
            .map(signed_posterior_to_color)
            .unwrap_or(Color::White);
        let reasoning_style = Some(Style::default().fg(segment_color));
        for content in &segment.content {
            match content {
                SegmentContent::Prompt(tokens) => {
                    push_text_as_spans(&mut lines, &tokens.decoded_string, None, &mut plain);
                }
                SegmentContent::ReasoningOrToolCall {
                    tokens,
                    complete: _,
                } => {
                    push_text_as_spans(
                        &mut lines,
                        &tokens.decoded_string,
                        reasoning_style,
                        &mut plain,
                    );
                }
                SegmentContent::ToolResponse(tokens) => {
                    push_text_as_spans(&mut lines, &tokens.decoded_string, None, &mut plain);
                }
            }
        }
    }
    if lines.is_empty() {
        lines.push(Vec::new());
    }
    let styled_lines: Vec<Line<'static>> = lines
        .into_iter()
        .map(|line_spans| {
            if line_spans.is_empty() {
                Line::from(String::new())
            } else {
                Line::from(line_spans)
            }
        })
        .collect();
    ConversationRender {
        plain,
        styled: Text::from(styled_lines),
    }
}

fn snapshot_for_model(
    model: LlmModelName,
    action_log: &DirectTreeActionLog,
    action_limit: usize,
    width_division_ratio: Option<usize>,
    override_hyperparameters: Option<PosteriorHyperparameters>,
) -> TreeSnapshot {
    let mut partial_log = action_log.clone();
    partial_log.actions = action_log
        .actions
        .iter()
        .take(action_limit)
        .cloned()
        .collect();
    match model {
        LlmModelName::Qwen25_7b => TreeSnapshot::from_tree(
            DirectTree::<Qwen25>::from_action_log(&partial_log),
            width_division_ratio,
            override_hyperparameters,
        ),
        LlmModelName::Qwen3_4b => TreeSnapshot::from_tree(
            DirectTree::<Qwen3_4B>::from_action_log(&partial_log),
            width_division_ratio,
            override_hyperparameters,
        ),
        LlmModelName::Qwen35_4b => TreeSnapshot::from_tree(
            DirectTree::<Qwen35_4B>::from_action_log(&partial_log),
            width_division_ratio,
            override_hyperparameters,
        ),
        LlmModelName::Gpt4o => TreeSnapshot::from_tree(
            DirectTree::<Gpt4o>::from_action_log(&partial_log),
            width_division_ratio,
            override_hyperparameters,
        ),
        LlmModelName::Gpt5Mini => TreeSnapshot::from_tree(
            DirectTree::<Gpt5Mini>::from_action_log(&partial_log),
            width_division_ratio,
            override_hyperparameters,
        ),
    }
}

fn width_division_ratio_for_model(model: LlmModelName, action_log: &DirectTreeActionLog) -> usize {
    match model {
        LlmModelName::Qwen25_7b => {
            width_division_ratio_for_tree(&DirectTree::<Qwen25>::from_action_log(action_log))
        }
        LlmModelName::Qwen3_4b => {
            width_division_ratio_for_tree(&DirectTree::<Qwen3_4B>::from_action_log(action_log))
        }
        LlmModelName::Qwen35_4b => {
            width_division_ratio_for_tree(&DirectTree::<Qwen35_4B>::from_action_log(action_log))
        }
        LlmModelName::Gpt4o => {
            width_division_ratio_for_tree(&DirectTree::<Gpt4o>::from_action_log(action_log))
        }
        LlmModelName::Gpt5Mini => {
            width_division_ratio_for_tree(&DirectTree::<Gpt5Mini>::from_action_log(action_log))
        }
    }
}

fn width_division_ratio_for_tree<M: credit_assignment::llm_model::LlmModelMarker>(
    tree: &DirectTree<M>,
) -> usize {
    let fixed_scale_ratio = fixed_width_scale_ratio_for_tree(tree);
    if fixed_scale_ratio <= 0.0 {
        return 1;
    }
    ((1.0 / fixed_scale_ratio).round() as usize).max(1)
}

fn question_stats_from_action_log(
    model: LlmModelName,
    action_log: &DirectTreeActionLog,
) -> (usize, usize, f64) {
    let final_snapshot =
        snapshot_for_model(model, action_log, action_log.actions.len(), None, None);
    let num_leaves = final_snapshot.leaf_segment_judgments.len();
    let num_correct = final_snapshot
        .leaf_segment_judgments
        .values()
        .filter(|judgment| judgment.is_correct)
        .count();
    let win_rate = if num_leaves == 0 {
        0.0
    } else {
        num_correct as f64 / num_leaves as f64
    };
    (num_correct, num_leaves, win_rate)
}

fn run_app(
    terminal: &mut Terminal<CrosstermBackend<Stdout>>,
    mut app: App,
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

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    std::panic::set_hook(Box::new(|info| {
        eprintln!("panic occurred: {}", info);
        let rust_backtrace = std::env::var("RUST_BACKTRACE").ok();
        if matches!(rust_backtrace.as_deref(), Some("1") | Some("full")) {
            let backtrace = Backtrace::force_capture();
            eprintln!("backtrace:\n{}", backtrace);
        }
        std::process::abort();
    }));
    let Args {
        model,
        config_nickname,
        rollout_config_path,
        posterior_hyperparameters_path,
        override_hyperparameters_path,
    } = Args::parse();
    let rollout_config: DirectRolloutConfig = read_json(rollout_config_path).unwrap();
    let posterior_hyperparameters =
        read_json::<PosteriorHyperparameters>(posterior_hyperparameters_path).unwrap();
    let override_hyperparameters = override_hyperparameters_path
        .map(|path| read_json::<PosteriorHyperparameters>(path).unwrap());
    let posterior_calculation_config = PosteriorCalculationConfig {
        hyperparameters: posterior_hyperparameters,
    };
    let asset_file_action_logs = AssetFileDirectTreeActionLogs {
        model,
        nickname: config_nickname,
        rollout_config,
        posterior_calculation_config,
    };
    let action_log_store = asset_file_action_logs.fetch().await;
    let mut keys = action_log_store.get_keys().await.unwrap();
    keys.sort();

    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;
    let app = App::new(model, action_log_store, keys, override_hyperparameters);
    let result = run_app(&mut terminal, app);
    disable_raw_mode()?;
    execute!(
        terminal.backend_mut(),
        LeaveAlternateScreen,
        DisableMouseCapture
    )?;
    terminal.show_cursor()?;
    result
}
