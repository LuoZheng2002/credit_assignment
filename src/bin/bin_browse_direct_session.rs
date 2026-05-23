use std::backtrace::Backtrace;
use std::collections::BTreeMap;
use std::error::Error;
use std::hash::{Hash, Hasher};
use std::io::{self, Stdout};

use clap::Parser;
use credit_assignment::{
    agent::{trajectory_action_types::FinalAnswer, tree::CorrectnessJudgment},
    direct_tool::{
        direct_tree::{DirectTree, Segment, SegmentContent, SegmentId},
        direct_tree_action_log::{AssetFileDirectTreeActionLogs, DirectTreeActionLog},
        posterior_calculation_config::{
            PosteriorCalculationConfig, PosteriorHyperparameters, TemperatureAccuracyPair,
        },
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
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, List, ListItem, ListState, Paragraph, Wrap};
use ratatui_core::buffer::Buffer;
use research_utility::asset_file::AssetFile;
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
    temperature_to_accuracy_path: String,
    #[arg(long)]
    posterior_hyperparameters_path: String,
}

const QUESTIONS_PER_PAGE: usize = 10;

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
}

impl TreeSnapshot {
    fn from_tree<M: credit_assignment::llm_model::LlmModelMarker>(tree: DirectTree<M>) -> Self {
        let root_segment_id = tree
            .root_segment_id
            .expect("Direct tree browser requires root segment");
        Self {
            segments: tree.segments,
            root_segment_id,
            leaf_segment_judgments: tree.leaf_segment_judgments,
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
    Conversation,
    Tree,
}

struct TreePage {
    entry_index: usize,
    total_actions: usize,
    action_limit: usize,
    snapshot: TreeSnapshot,
    selected_segment_id: SegmentId,
    segment_order: Vec<SegmentId>,
    tree_lines: Vec<String>,
    rendered_segments: Vec<TreeRenderedSegment>,
    hovered_segment_id: Option<SegmentId>,
    tree_area: Option<Rect>,
}

#[derive(Clone, Copy)]
struct TreeRenderedSegment {
    segment_id: SegmentId,
    row: usize,
    col: usize,
}

impl TreePage {
    fn new(model: LlmModelName, entry_index: usize, entry: &QuestionEntry) -> Self {
        let total_actions = entry.action_log.actions.len();
        let action_limit = total_actions;
        let snapshot = snapshot_for_model(model, &entry.action_log, action_limit);
        let selected_segment_id = snapshot.root_segment_id;
        let (tree_lines, rendered_segments) = build_segment_graph_lines(&snapshot);
        let segment_order = collect_segment_preorder(&snapshot);
        Self {
            entry_index,
            total_actions,
            action_limit,
            snapshot,
            selected_segment_id,
            segment_order,
            tree_lines,
            rendered_segments,
            hovered_segment_id: None,
            tree_area: None,
        }
    }

    fn selected_row_index(&self) -> Option<usize> {
        self.rendered_segments
            .iter()
            .find(|segment| segment.segment_id == self.selected_segment_id)
            .map(|segment| segment.row)
    }

    fn move_selected_segment_by(&mut self, delta: isize) {
        let Some(current_index) = self
            .segment_order
            .iter()
            .position(|id| *id == self.selected_segment_id)
        else {
            return;
        };
        let total = self.segment_order.len();
        if total == 0 {
            return;
        }
        let next = (current_index as isize + delta).clamp(0, total as isize - 1) as usize;
        self.selected_segment_id = self.segment_order[next];
    }

    fn rebuild_snapshot(&mut self, model: LlmModelName, entry: &QuestionEntry) {
        self.snapshot = snapshot_for_model(model, &entry.action_log, self.action_limit);
        let (tree_lines, rendered_segments) = build_segment_graph_lines(&self.snapshot);
        self.segment_order = collect_segment_preorder(&self.snapshot);
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

    fn set_action_limit(&mut self, model: LlmModelName, entry: &QuestionEntry, new_limit: usize) {
        if self.action_limit == new_limit {
            return;
        }
        self.action_limit = new_limit;
        self.rebuild_snapshot(model, entry);
    }
}

struct App {
    model: LlmModelName,
    entries: Vec<QuestionEntry>,
    mode: Mode,
    home_selected_index: usize,
    home_list_area: Option<Rect>,
    conversation_area: Option<Rect>,
    tree_page: Option<TreePage>,
    tree_focus: TreePaneFocus,
    conversation_scroll: usize,
    conversation_max_scroll: usize,
    conversation_metrics: Option<PaneMetrics>,
}

impl App {
    fn new(model: LlmModelName, entries: Vec<QuestionEntry>) -> Self {
        Self {
            model,
            entries,
            mode: Mode::Home,
            home_selected_index: 0,
            home_list_area: None,
            conversation_area: None,
            tree_page: None,
            tree_focus: TreePaneFocus::Tree,
            conversation_scroll: 0,
            conversation_max_scroll: 0,
            conversation_metrics: None,
        }
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

        let total_pages = if self.entries.is_empty() {
            1
        } else {
            self.entries.len().div_ceil(QUESTIONS_PER_PAGE)
        };
        let current_page = if self.entries.is_empty() {
            1
        } else {
            self.home_selected_index / QUESTIONS_PER_PAGE + 1
        };

        let title = Paragraph::new(format!(
            "Questions and win rate (page {current_page}/{total_pages}, total {})",
            self.entries.len()
        ))
        .block(Block::default().borders(Borders::ALL).title("Home"));
        frame.render_widget(title, chunks[0]);

        if self.entries.is_empty() {
            let empty = Paragraph::new("No direct session logs found")
                .block(Block::default().borders(Borders::ALL).title("Questions"));
            frame.render_widget(empty, chunks[1]);
        } else {
            let page_start = (self.home_selected_index / QUESTIONS_PER_PAGE) * QUESTIONS_PER_PAGE;
            let page_end = (page_start + QUESTIONS_PER_PAGE).min(self.entries.len());
            let page_entries = &self.entries[page_start..page_end];

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
        let Some(tree_page) = self.tree_page.as_mut() else {
            return;
        };
        let entry = &self.entries[tree_page.entry_index];
        let selected_segment = tree_page
            .snapshot
            .segments
            .get(&tree_page.selected_segment_id)
            .expect("selected segment must exist");

        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(8),
                Constraint::Min(8),
                Constraint::Length(10),
            ])
            .split(frame.area());

        let judgment = tree_page
            .snapshot
            .leaf_segment_judgments
            .get(&tree_page.selected_segment_id);

        let mut summary = format!(
            "Question #{}\nQuestion: {}\nCorrect answer: {}\nActions applied: {}/{}\nSelected segment: S{} (children: {})",
            entry.key,
            entry.action_log.question.question,
            entry.action_log.question.correct_answer,
            tree_page.action_limit,
            tree_page.total_actions,
            tree_page.selected_segment_id.0,
            selected_segment.child_ids.len()
        );
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

        frame.render_widget(
            Paragraph::new(summary)
                .wrap(Wrap { trim: false })
                .block(Block::default().borders(Borders::ALL).title("Summary")),
            chunks[0],
        );

        let conversation =
            build_conversation_text(&tree_page.snapshot, tree_page.selected_segment_id);
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
            &conversation,
            conversation_inner,
            &mut self.conversation_metrics,
        );
        let conversation_max_scroll =
            bottom_scroll_limit(conversation_lines, conversation_height.max(1));
        self.conversation_max_scroll = conversation_max_scroll;
        frame.render_widget(
            Paragraph::new(conversation)
                .wrap(Wrap { trim: false })
                .block(conversation_block)
                .scroll((clamp_scroll(self.conversation_scroll), 0)),
            chunks[1],
        );

        tree_page.tree_area = Some(chunks[2]);
        let list_items: Vec<ListItem> = (0..tree_page.tree_lines.len())
            .map(|row| ListItem::new(render_tree_line(tree_page, row)))
            .collect();
        let mut state = ListState::default();
        state.select(tree_page.selected_row_index());
        let tree_block = Block::default()
            .borders(Borders::ALL)
            .title(
                "Segment tree (Left/Right changes action count, click or Up/Down selects segment)",
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
                .highlight_style(Style::default().bg(Color::DarkGray))
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
                if !self.entries.is_empty() {
                    self.home_selected_index =
                        (self.home_selected_index + 1).min(self.entries.len() - 1);
                }
                false
            }
            KeyCode::Left => {
                self.home_selected_index =
                    self.home_selected_index.saturating_sub(QUESTIONS_PER_PAGE);
                false
            }
            KeyCode::Right => {
                if !self.entries.is_empty() {
                    self.home_selected_index =
                        (self.home_selected_index + QUESTIONS_PER_PAGE).min(self.entries.len() - 1);
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
        let Some(tree_page) = self.tree_page.as_mut() else {
            return false;
        };
        let entry = &self.entries[tree_page.entry_index];
        match key.code {
            KeyCode::Char('q') | KeyCode::Esc => {
                self.mode = Mode::Home;
                self.tree_page = None;
                self.conversation_area = None;
                self.conversation_scroll = 0;
                self.conversation_max_scroll = 0;
                self.conversation_metrics = None;
                false
            }
            KeyCode::Left => {
                let next = tree_page.action_limit.saturating_sub(1);
                tree_page.set_action_limit(self.model, entry, next);
                false
            }
            KeyCode::Right => {
                let next = (tree_page.action_limit + 1).min(tree_page.total_actions);
                tree_page.set_action_limit(self.model, entry, next);
                false
            }
            KeyCode::Up => {
                if self.tree_focus == TreePaneFocus::Conversation {
                    self.scroll_conversation_up(1);
                } else {
                    tree_page.move_selected_segment_by(-1);
                }
                false
            }
            KeyCode::Down => {
                if self.tree_focus == TreePaneFocus::Conversation {
                    self.conversation_scroll = self.conversation_scroll.saturating_add(1);
                } else {
                    tree_page.move_selected_segment_by(1);
                }
                false
            }
            KeyCode::PageUp => {
                if self.tree_focus == TreePaneFocus::Conversation {
                    self.scroll_conversation_up(10);
                } else {
                    tree_page.move_selected_segment_by(-10);
                }
                false
            }
            KeyCode::PageDown => {
                if self.tree_focus == TreePaneFocus::Conversation {
                    self.conversation_scroll = self.conversation_scroll.saturating_add(10);
                } else {
                    tree_page.move_selected_segment_by(10);
                }
                false
            }
            KeyCode::Home => {
                if self.tree_focus == TreePaneFocus::Conversation {
                    self.conversation_scroll = 0;
                }
                false
            }
            KeyCode::Tab => {
                self.tree_focus = match self.tree_focus {
                    TreePaneFocus::Conversation => TreePaneFocus::Tree,
                    TreePaneFocus::Tree => TreePaneFocus::Conversation,
                };
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
        let Some(tree_page) = self.tree_page.as_mut() else {
            return;
        };
        if let Some(conversation_area) = self.conversation_area {
            if contains_point(conversation_area, mouse.column, mouse.row) {
                self.tree_focus = TreePaneFocus::Conversation;
            } else if let Some(tree_area) = tree_page.tree_area {
                if contains_point(tree_area, mouse.column, mouse.row) {
                    self.tree_focus = TreePaneFocus::Tree;
                }
            }
        }

        let hovered = tree_segment_at_mouse(tree_page, mouse.column, mouse.row);
        match mouse.kind {
            MouseEventKind::Moved => {
                tree_page.hovered_segment_id = hovered;
            }
            MouseEventKind::Down(MouseButton::Left) => {
                tree_page.hovered_segment_id = hovered;
                if let Some(segment_id) = hovered {
                    tree_page.selected_segment_id = segment_id;
                }
            }
            MouseEventKind::ScrollUp => {
                if self.tree_focus == TreePaneFocus::Conversation {
                    self.scroll_conversation_up(1);
                } else if self.tree_focus == TreePaneFocus::Tree {
                    let entry = &self.entries[tree_page.entry_index];
                    let next = tree_page.action_limit.saturating_sub(1);
                    tree_page.set_action_limit(self.model, entry, next);
                }
            }
            MouseEventKind::ScrollDown => {
                if self.tree_focus == TreePaneFocus::Conversation {
                    self.conversation_scroll = self.conversation_scroll.saturating_add(1);
                } else if self.tree_focus == TreePaneFocus::Tree {
                    let entry = &self.entries[tree_page.entry_index];
                    let next = (tree_page.action_limit + 1).min(tree_page.total_actions);
                    tree_page.set_action_limit(self.model, entry, next);
                }
            }
            _ => {}
        }
    }

    fn open_selected_home_entry(&mut self) {
        if self.entries.is_empty() {
            return;
        }
        let entry = &self.entries[self.home_selected_index];
        self.tree_page = Some(TreePage::new(self.model, self.home_selected_index, entry));
        self.mode = Mode::Tree;
        self.tree_focus = TreePaneFocus::Tree;
        self.conversation_scroll = 0;
        self.conversation_max_scroll = 0;
        self.conversation_metrics = None;
        self.conversation_area = None;
    }

    fn scroll_conversation_up(&mut self, magnitude: usize) {
        if self.conversation_scroll > self.conversation_max_scroll {
            self.conversation_scroll = self.conversation_max_scroll;
            return;
        }
        self.conversation_scroll = self.conversation_scroll.saturating_sub(magnitude);
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
        let page_end = (page_start + QUESTIONS_PER_PAGE).min(self.entries.len());
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

fn collect_segment_preorder(snapshot: &TreeSnapshot) -> Vec<SegmentId> {
    let mut order = Vec::new();
    let mut stack = vec![snapshot.root_segment_id];
    while let Some(segment_id) = stack.pop() {
        order.push(segment_id);
        if let Some(segment) = snapshot.segments.get(&segment_id) {
            for child in segment.child_ids.iter().rev() {
                stack.push(*child);
            }
        }
    }
    order
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

fn segment_depth(snapshot: &TreeSnapshot, segment_id: SegmentId) -> usize {
    path_root_to_segment(snapshot, segment_id)
        .len()
        .saturating_sub(1)
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

    let mut col_by_segment: BTreeMap<SegmentId, usize> = BTreeMap::new();
    for segment_id in snapshot.segments.keys().copied() {
        let depth = segment_depth(snapshot, segment_id);
        col_by_segment.insert(segment_id, depth * 8);
    }

    let mut rendered_segments = Vec::new();
    for segment_id in snapshot.segments.keys().copied() {
        let Some(&row) = row_by_segment.get(&segment_id) else {
            continue;
        };
        let col = *col_by_segment
            .get(&segment_id)
            .expect("segment col must be available");
        write_pattern(&mut canvas, row, col, "=====");
        rendered_segments.push(TreeRenderedSegment {
            segment_id,
            row,
            col,
        });
    }

    for (parent_id, parent) in &snapshot.segments {
        let Some(&parent_row) = row_by_segment.get(parent_id) else {
            continue;
        };
        let parent_col = *col_by_segment
            .get(parent_id)
            .expect("parent col must be available");
        let edge_col = parent_col + 5;

        for child_id in parent.child_ids.iter().copied() {
            let child_row = *row_by_segment
                .get(&child_id)
                .expect("child row must be available");
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
                if has_lower_sibling {
                    write_pattern(&mut canvas, parent_row, edge_col, "━┳━");
                } else {
                    write_pattern(&mut canvas, parent_row, edge_col, "━━━");
                }
            } else {
                for row in (parent_row + 1)..child_row {
                    write_pattern(&mut canvas, row, edge_col, " ┃ ");
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
                let connector = if has_lower_sibling {
                    " ┣━"
                } else {
                    " ┗━"
                };
                write_pattern(&mut canvas, child_row, edge_col, connector);
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
            + 5;
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

fn render_tree_line(tree_page: &TreePage, row: usize) -> Line<'static> {
    let line = tree_page
        .tree_lines
        .get(row)
        .expect("tree row must be in bounds");
    let line_chars: Vec<char> = line.chars().collect();
    let mut styles: Vec<Option<Style>> = vec![None; line_chars.len()];

    for rendered in tree_page.rendered_segments.iter().copied() {
        if rendered.row != row {
            continue;
        }
        let style = if rendered.segment_id == tree_page.selected_segment_id {
            Some(
                Style::default()
                    .fg(Color::LightGreen)
                    .add_modifier(Modifier::BOLD),
            )
        } else if tree_page.hovered_segment_id == Some(rendered.segment_id) {
            Some(
                Style::default()
                    .fg(Color::Yellow)
                    .add_modifier(Modifier::BOLD),
            )
        } else {
            None
        };
        if let Some(style) = style {
            for idx in rendered.col..(rendered.col + 5) {
                if idx < styles.len() {
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
    let mut segment_style = styles.first().copied().flatten();
    for idx in 0..line_chars.len() {
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

fn tree_segment_at_mouse(tree_page: &TreePage, column: u16, row: u16) -> Option<SegmentId> {
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
    let local_col = (column - area.x - 1) as usize;
    tree_page
        .rendered_segments
        .iter()
        .copied()
        .find(|rendered| {
            rendered.row == local_row && local_col >= rendered.col && local_col < rendered.col + 5
        })
        .map(|rendered| rendered.segment_id)
}

fn build_conversation_text(snapshot: &TreeSnapshot, segment_id: SegmentId) -> String {
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

    let mut out = String::new();
    for sid in path {
        let segment = snapshot
            .segments
            .get(&sid)
            .expect("segment in path must exist");
        out.push_str(&format!("=== Segment S{} ===\n", sid.0));
        for content in &segment.content {
            match content {
                SegmentContent::Prompt(tokens) => {
                    out.push_str("[Prompt]\n");
                    out.push_str(&tokens.decoded_string);
                    out.push_str("\n\n");
                }
                SegmentContent::ReasoningOrToolCall { tokens, complete } => {
                    out.push_str(&format!("[Reasoning complete={complete}]\n"));
                    out.push_str(&tokens.decoded_string);
                    out.push_str("\n\n");
                }
                SegmentContent::ToolResponse(tokens) => {
                    out.push_str("[Tool response]\n");
                    out.push_str(&tokens.decoded_string);
                    out.push_str("\n\n");
                }
            }
        }
    }
    out
}

fn snapshot_for_model(
    model: LlmModelName,
    action_log: &DirectTreeActionLog,
    action_limit: usize,
) -> TreeSnapshot {
    let mut partial_log = action_log.clone();
    partial_log.actions = action_log
        .actions
        .iter()
        .take(action_limit)
        .cloned()
        .collect();
    match model {
        LlmModelName::Qwen25_7b => {
            TreeSnapshot::from_tree(DirectTree::<Qwen25>::from_action_log(&partial_log))
        }
        LlmModelName::Qwen3_4b => {
            TreeSnapshot::from_tree(DirectTree::<Qwen3_4B>::from_action_log(&partial_log))
        }
        LlmModelName::Qwen35_4b => {
            TreeSnapshot::from_tree(DirectTree::<Qwen35_4B>::from_action_log(&partial_log))
        }
        LlmModelName::Gpt4o => {
            TreeSnapshot::from_tree(DirectTree::<Gpt4o>::from_action_log(&partial_log))
        }
        LlmModelName::Gpt5Mini => {
            TreeSnapshot::from_tree(DirectTree::<Gpt5Mini>::from_action_log(&partial_log))
        }
    }
}

fn question_stats_from_action_log(
    model: LlmModelName,
    action_log: &DirectTreeActionLog,
) -> (usize, usize, f64) {
    let final_snapshot = snapshot_for_model(model, action_log, action_log.actions.len());
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
        temperature_to_accuracy_path,
        posterior_hyperparameters_path,
    } = Args::parse();
    let rollout_config = read_json(rollout_config_path).unwrap();
    let temperature_to_accuracy =
        read_json::<Vec<TemperatureAccuracyPair>>(temperature_to_accuracy_path).unwrap();
    let posterior_hyperparameters =
        read_json::<PosteriorHyperparameters>(posterior_hyperparameters_path).unwrap();
    let posterior_calculation_config = PosteriorCalculationConfig {
        temperature_to_accuracy,
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

    let mut entries = Vec::with_capacity(keys.len());
    for key in keys {
        let action_log = action_log_store
            .get(key)
            .await
            .unwrap()
            .expect("key from sqlite key set must exist");
        let (num_correct, num_leaves, win_rate) =
            question_stats_from_action_log(model, &action_log);
        entries.push(QuestionEntry {
            key,
            action_log,
            win_rate,
            num_correct,
            num_leaves,
        });
    }

    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;
    let app = App::new(model, entries);
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
