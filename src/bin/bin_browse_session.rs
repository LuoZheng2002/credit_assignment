use std::collections::{HashMap, HashSet};
use std::error::Error;
use std::fs::File;
use std::io::{self, BufRead, BufReader, Stdout};
use std::path::PathBuf;

use clap::Parser;
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
use ratatui::widgets::{Block, Borders, List, ListItem, ListState, Paragraph, Wrap};
use ratatui_core::buffer::Buffer;
use serde_json;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use credit_assignment::deepmath::judge_answers::DeepMathCorrectness;
use credit_assignment::multi_agent::generate_rollout_answers::RolloutTrajectory;
use credit_assignment::multi_agent::rollout::get_prompt_according_to_session_status;
use credit_assignment::multi_agent::session::{
    NextStepDecision, RolloutAction, TrajectoryActionLog, TrajectoryState, Tree,
};

/// Command line arguments for browsing rollout session logs.
#[derive(Parser, Debug)]
#[command(author, version, about = "Interactively browse rollout session logs")]
struct Args {
    #[arg(
        short = 'f',
        long = "file",
        help = "Path to a jsonl file containing RolloutAnswerRaw lines"
    )]
    file: PathBuf,
    #[arg(
        long = "correctness-file",
        help = "Path to a jsonl file containing DeepMathCorrectness lines"
    )]
    correctness_file: Option<PathBuf>,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();
    let answers = load_rollout_answers(&args.file)?;
    let correctness_by_id = match &args.correctness_file {
        Some(path) => Some(load_correctness_by_id(path)?),
        None => None,
    };
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;
    let result = run_app(&mut terminal, answers, correctness_by_id);
    disable_raw_mode()?;
    execute!(
        terminal.backend_mut(),
        LeaveAlternateScreen,
        DisableMouseCapture
    )?;
    terminal.show_cursor()?;
    result
}

fn load_correctness_by_id(path: &PathBuf) -> Result<HashMap<usize, bool>, Box<dyn Error>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut correctness_by_id: HashMap<usize, bool> = HashMap::new();
    for line in reader.lines() {
        let line = line?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let correctness: DeepMathCorrectness = serde_json::from_str(trimmed)?;
        correctness_by_id.insert(correctness.id, correctness.correct);
    }
    Ok(correctness_by_id)
}

fn load_rollout_answers(path: &PathBuf) -> Result<Vec<RolloutTrajectory>, Box<dyn Error>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut answers = Vec::new();
    for line in reader.lines() {
        let line = line?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let answer: RolloutTrajectory = serde_json::from_str(trimmed)?;
        answers.push(answer);
    }
    if answers.is_empty() {
        Err("The provided jsonl file does not contain any RolloutAnswerRaw entries".into())
    } else {
        answers.sort_by_key(|answer| answer.id);
        Ok(answers)
    }
}

fn run_app(
    terminal: &mut Terminal<CrosstermBackend<Stdout>>,
    answers: Vec<RolloutTrajectory>,
    correctness_by_id: Option<HashMap<usize, bool>>,
) -> Result<(), Box<dyn Error>> {
    let mut app = App::new(answers, correctness_by_id);
    loop {
        terminal.draw(|f| app.draw(f))?;
        match event::read()? {
            Event::Key(key) => {
                if app.handle_key(key) {
                    break;
                }
            }
            Event::Mouse(mouse) => {
                if app.handle_mouse(mouse) {
                    break;
                }
            }
            _ => {}
        }
    }
    Ok(())
}

enum SelectionAction {
    Continue,
    Quit,
    OpenSession,
}

enum BrowsingAction {
    Continue,
    GoBack,
}

enum TreeAction {
    Continue,
    GoBack,
    OpenSession,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
enum PaneFocus {
    Log,
    Prompt,
    Action,
}

impl PaneFocus {
    fn next(self) -> Self {
        match self {
            PaneFocus::Log => PaneFocus::Prompt,
            PaneFocus::Prompt => PaneFocus::Action,
            PaneFocus::Action => PaneFocus::Log,
        }
    }

    fn prev(self) -> Self {
        match self {
            PaneFocus::Log => PaneFocus::Action,
            PaneFocus::Prompt => PaneFocus::Log,
            PaneFocus::Action => PaneFocus::Prompt,
        }
    }
}

struct App {
    answers: Vec<RolloutTrajectory>,
    correctness_by_id: Option<HashMap<usize, bool>>,
    selection_state: ListState,
    tree_view: Option<TreeView>,
    browsing_view: Option<SessionView>,
    focus: PaneFocus,
    tree_horizontal_scroll: usize,
    tree_hovered_line_index: Option<usize>,
    prompt_scroll: usize,
    action_scroll: usize,
    log_area: Option<Rect>,
    prompt_area: Option<Rect>,
    action_area: Option<Rect>,
    selection_area: Option<Rect>,
    prompt_metrics: Option<PaneMetrics>,
    action_metrics: Option<PaneMetrics>,
}

impl App {
    fn new(
        answers: Vec<RolloutTrajectory>,
        correctness_by_id: Option<HashMap<usize, bool>>,
    ) -> Self {
        let mut selection_state = ListState::default();
        if !answers.is_empty() {
            selection_state.select(Some(0));
        }
        Self {
            answers,
            correctness_by_id,
            selection_state,
            tree_view: None,
            browsing_view: None,
            focus: PaneFocus::Log,
            tree_horizontal_scroll: 0,
            tree_hovered_line_index: None,
            prompt_scroll: 0,
            action_scroll: 0,
            log_area: None,
            prompt_area: None,
            action_area: None,
            selection_area: None,
            prompt_metrics: None,
            action_metrics: None,
        }
    }

    fn draw(&mut self, frame: &mut ratatui::Frame<'_>) {
        if self.browsing_view.is_some() {
            self.draw_session(frame);
        } else if self.tree_view.is_some() {
            self.draw_tree(frame);
        } else {
            self.draw_selection(frame);
        }
    }

    fn handle_key(&mut self, key: KeyEvent) -> bool {
        if self.browsing_view.is_some() {
            match self.handle_session_key(key) {
                BrowsingAction::Continue => false,
                BrowsingAction::GoBack => {
                    self.browsing_view = None;
                    assert!(self.tree_view.is_some(), "Session go-back requires active tree view");
                    self.tree_horizontal_scroll = 0;
                    self.tree_hovered_line_index = None;
                    false
                }
            }
        } else if self.tree_view.is_some() {
            match self.handle_tree_key(key) {
                TreeAction::Continue => false,
                TreeAction::GoBack => {
                    self.tree_view = None;
                    self.tree_horizontal_scroll = 0;
                    self.tree_hovered_line_index = None;
                    false
                }
                TreeAction::OpenSession => {
                    self.open_session_for_selected_tree_node();
                    false
                }
            }
        } else {
            match self.handle_selection_key(key) {
                SelectionAction::Continue => false,
                SelectionAction::Quit => true,
                SelectionAction::OpenSession => {
                    self.open_selected_question();
                    false
                }
            }
        }
    }

    fn handle_mouse(&mut self, mouse_event: MouseEvent) -> bool {
        if self.browsing_view.is_some() {
            match mouse_event.kind {
                MouseEventKind::Moved => {
                    self.update_focus_from_mouse(mouse_event.column, mouse_event.row);
                    false
                }
                MouseEventKind::Down(MouseButton::Left) => {
                    self.update_focus_from_mouse(mouse_event.column, mouse_event.row);
                    false
                }
                MouseEventKind::ScrollUp => {
                    self.update_focus_from_mouse(mouse_event.column, mouse_event.row);
                    self.apply_mouse_scroll(-1);
                    false
                }
                MouseEventKind::ScrollDown => {
                    self.update_focus_from_mouse(mouse_event.column, mouse_event.row);
                    self.apply_mouse_scroll(1);
                    false
                }
                _ => false,
            }
        } else if self.tree_view.is_some() {
            match mouse_event.kind {
                MouseEventKind::Moved => {
                    self.tree_hovered_line_index =
                        self.tree_line_index_from_mouse(mouse_event.column, mouse_event.row);
                    if let Some(hovered_index) = self.tree_hovered_line_index {
                        let view = self.tree_view.as_mut().unwrap();
                        if hovered_index < view.tree_lines.len() {
                            view.select_tree_line_by_index(hovered_index);
                        }
                    }
                    false
                }
                MouseEventKind::Down(MouseButton::Left) => {
                    if let Some(local_index) = self
                        .tree_line_index_from_mouse(mouse_event.column, mouse_event.row)
                    {
                        {
                            let view = self.tree_view.as_mut().unwrap();
                            if local_index < view.tree_lines.len() {
                                view.select_tree_line_by_index(local_index);
                            }
                        }
                        self.open_session_for_selected_tree_node();
                    }
                    false
                }
                MouseEventKind::ScrollUp => {
                    if let Some(view) = self.tree_view.as_mut() {
                        view.move_tree_selection_by(-1);
                    }
                    false
                }
                MouseEventKind::ScrollDown => {
                    if let Some(view) = self.tree_view.as_mut() {
                        view.move_tree_selection_by(1);
                    }
                    false
                }
                _ => false,
            }
        } else {
            match mouse_event.kind {
                MouseEventKind::Moved => {
                    self.update_selection_from_mouse(mouse_event.column, mouse_event.row);
                    false
                }
                MouseEventKind::Down(MouseButton::Left) => {
                    self.update_selection_from_mouse(mouse_event.column, mouse_event.row);
                    self.open_selected_question();
                    false
                }
                MouseEventKind::ScrollUp => {
                    self.scroll_selection_window(-1);
                    false
                }
                MouseEventKind::ScrollDown => {
                    self.scroll_selection_window(1);
                    false
                }
                _ => false,
            }
        }
    }

    fn update_selection_from_mouse(&mut self, column: u16, row: u16) {
        if let Some(index) = self.selection_index_from_mouse(column, row) {
            self.selection_state.select(Some(index));
        }
    }

    fn selection_index_from_mouse(&self, column: u16, row: u16) -> Option<usize> {
        let rect = self.selection_area?;
        if !contains_point(rect, column, row) {
            return None;
        }
        let inner_height = rect.height.saturating_sub(2);
        if inner_height == 0 {
            return None;
        }
        if row <= rect.y || row >= rect.y + rect.height - 1 {
            return None;
        }
        let local_row = row - rect.y - 1;
        if local_row >= inner_height {
            return None;
        }
        let offset = self.selection_state.offset();
        offset.checked_add(local_row as usize)
    }

    fn scroll_selection_window(&mut self, delta: isize) {
        if self.answers.is_empty() {
            return;
        }
        let current_offset = self.selection_state.offset();
        let next_offset = if delta < 0 {
            current_offset.saturating_sub(delta.unsigned_abs())
        } else {
            current_offset.saturating_add(delta as usize)
        };
        let max_offset = self.answers.len().saturating_sub(1);
        *self.selection_state.offset_mut() = next_offset.min(max_offset);
    }

    fn open_selected_question(&mut self) {
        if let Some(selected) = self.selection_state.selected() {
            if selected < self.answers.len() {
                let answer = self.answers[selected].clone();
                self.tree_view = Some(TreeView::new(answer));
                self.focus = PaneFocus::Log;
                self.tree_horizontal_scroll = 0;
                self.tree_hovered_line_index = None;
                self.prompt_scroll = 0;
                self.action_scroll = 0;
                self.selection_area = None;
            }
        }
    }

    fn open_session_for_selected_tree_node(&mut self) {
        let tree_view = self
            .tree_view
            .as_ref()
            .expect("Tree page must exist before opening session page");
        let selected_node_id = tree_view.selected_node_id();
        let answer = tree_view.answer.clone();
        self.browsing_view = Some(SessionView::new(answer, selected_node_id));
        self.focus = PaneFocus::Log;
        self.tree_hovered_line_index = None;
        self.prompt_scroll = 0;
        self.action_scroll = 0;
    }

    fn handle_selection_key(&mut self, key: KeyEvent) -> SelectionAction {
        match key.code {
            KeyCode::Enter => {
                if self.selection_state.selected().is_some() {
                    return SelectionAction::OpenSession;
                }
                SelectionAction::Continue
            }
            KeyCode::Char('q') | KeyCode::Esc => SelectionAction::Quit,
            _ => SelectionAction::Continue,
        }
    }

    fn handle_tree_key(&mut self, key: KeyEvent) -> TreeAction {
        let mut new_tree_horizontal_scroll = self.tree_horizontal_scroll;
        let action = {
            let view = self.tree_view.as_mut().unwrap();
            match key.code {
                KeyCode::Left => {
                    new_tree_horizontal_scroll = new_tree_horizontal_scroll.saturating_sub(1);
                    TreeAction::Continue
                }
                KeyCode::Right => {
                    new_tree_horizontal_scroll = new_tree_horizontal_scroll.saturating_add(1);
                    TreeAction::Continue
                }
                KeyCode::Up => {
                    view.move_tree_selection_by(-1);
                    TreeAction::Continue
                }
                KeyCode::Down => {
                    view.move_tree_selection_by(1);
                    TreeAction::Continue
                }
                KeyCode::Home => {
                    view.move_tree_selection_to_start();
                    TreeAction::Continue
                }
                KeyCode::End => {
                    view.move_tree_selection_to_end();
                    TreeAction::Continue
                }
                KeyCode::PageUp => {
                    view.move_tree_selection_by(-10);
                    TreeAction::Continue
                }
                KeyCode::PageDown => {
                    view.move_tree_selection_by(10);
                    TreeAction::Continue
                }
                KeyCode::Enter => TreeAction::OpenSession,
                KeyCode::Esc => TreeAction::GoBack,
                KeyCode::Char('q') => TreeAction::GoBack,
                _ => TreeAction::Continue,
            }
        };
        self.tree_horizontal_scroll = new_tree_horizontal_scroll;
        action
    }

    fn handle_session_key(&mut self, key: KeyEvent) -> BrowsingAction {
        let mut new_focus = self.focus;
        let mut new_prompt_scroll = self.prompt_scroll;
        let mut new_action_scroll = self.action_scroll;
        let action = {
            let view = self.browsing_view.as_mut().unwrap();
            match key.code {
                KeyCode::Left => {
                    new_focus = new_focus.prev();
                    BrowsingAction::Continue
                }
                KeyCode::Right => {
                    new_focus = new_focus.next();
                    BrowsingAction::Continue
                }
                KeyCode::Tab => {
                    new_focus = new_focus.next();
                    BrowsingAction::Continue
                }
                KeyCode::BackTab => {
                    new_focus = new_focus.prev();
                    BrowsingAction::Continue
                }
                KeyCode::Up => {
                    match new_focus {
                        PaneFocus::Log => view.move_by(-1),
                        PaneFocus::Prompt => {
                            new_prompt_scroll = new_prompt_scroll.saturating_sub(1);
                        }
                        PaneFocus::Action => {
                            new_action_scroll = new_action_scroll.saturating_sub(1);
                        }
                    }
                    BrowsingAction::Continue
                }
                KeyCode::Down => {
                    match new_focus {
                        PaneFocus::Log => view.move_by(1),
                        PaneFocus::Prompt => {
                            new_prompt_scroll = new_prompt_scroll.saturating_add(1);
                        }
                        PaneFocus::Action => {
                            new_action_scroll = new_action_scroll.saturating_add(1);
                        }
                    }
                    BrowsingAction::Continue
                }
                KeyCode::Home => {
                    if new_focus == PaneFocus::Log { view.move_to_start(); }
                    BrowsingAction::Continue
                }
                KeyCode::End => {
                    if new_focus == PaneFocus::Log { view.move_to_end(); }
                    BrowsingAction::Continue
                }
                KeyCode::PageUp => {
                    if new_focus == PaneFocus::Log { view.move_by(-10); }
                    BrowsingAction::Continue
                }
                KeyCode::PageDown => {
                    if new_focus == PaneFocus::Log { view.move_by(10); }
                    BrowsingAction::Continue
                }
                KeyCode::Esc | KeyCode::Char('q') => BrowsingAction::GoBack,
                _ => BrowsingAction::Continue,
            }
        };
        self.focus = new_focus;
        self.prompt_scroll = new_prompt_scroll;
        self.action_scroll = new_action_scroll;
        action
    }

    fn update_focus_from_mouse(&mut self, column: u16, row: u16) {
        if self.browsing_view.is_none() {
            return;
        }
        if let Some(rect) = self.log_area {
            if contains_point(rect, column, row) {
                self.focus = PaneFocus::Log;
                return;
            }
        }
        if let Some(rect) = self.prompt_area {
            if contains_point(rect, column, row) {
                self.focus = PaneFocus::Prompt;
                return;
            }
        }
        if let Some(rect) = self.action_area {
            if contains_point(rect, column, row) {
                self.focus = PaneFocus::Action;
            }
        }
    }

    fn apply_mouse_scroll(&mut self, delta: isize) {
        if self.browsing_view.is_none() {
            return;
        }
        let view = self.browsing_view.as_mut().unwrap();
        match self.focus {
            PaneFocus::Log => view.move_by(delta),
            PaneFocus::Prompt => adjust_scroll_offset(&mut self.prompt_scroll, delta),
            PaneFocus::Action => adjust_scroll_offset(&mut self.action_scroll, delta),
        }
    }

    fn draw_tree(&mut self, frame: &mut ratatui::Frame<'_>) {
        let view = self.tree_view.as_ref().unwrap();
        self.selection_area = None;
        self.log_area = Some(frame.area());
        self.prompt_area = None;
        self.action_area = None;

        let title = format!(
            "Tree view for question {} (selected leaf: {}, hscroll: {})",
            view.answer.id,
            view.selected_node_id(),
            self.tree_horizontal_scroll,
        );
        let block = Block::default().borders(Borders::ALL).title(title);
        let items: Vec<ListItem> = view
            .tree_lines
            .iter()
            .enumerate()
            .map(|(i, line)| {
                let mut item = ListItem::new(slice_line_from_char_offset(
                    line,
                    self.tree_horizontal_scroll,
                ));
                if self.tree_hovered_line_index == Some(i) {
                    item = item.style(Style::default().bg(Color::DarkGray));
                }
                item
            })
            .collect();
        let mut state = ListState::default();
        state.select(Some(view.selected_tree_line_index));
        frame.render_stateful_widget(
            List::new(items)
                .block(block)
                .highlight_style(Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD))
                .highlight_symbol("▶ "),
            frame.area(),
            &mut state,
        );
    }

    fn tree_line_index_from_mouse(&self, column: u16, row: u16) -> Option<usize> {
        let rect = self.log_area?;
        if !contains_point(rect, column, row) {
            return None;
        }
        let inner_height = rect.height.saturating_sub(2);
        if inner_height == 0 {
            return None;
        }
        if row <= rect.y || row >= rect.y + rect.height - 1 {
            return None;
        }
        let local_row = row - rect.y - 1;
        Some(local_row as usize)
    }

    fn draw_selection(&mut self, frame: &mut ratatui::Frame<'_>) {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(3),
                Constraint::Min(4),
                Constraint::Length(3),
            ])
            .split(frame.area());

        let header = Paragraph::new("Select a question to inspect its session log").block(
            Block::default()
                .borders(Borders::ALL)
                .title("Rollout overview"),
        );
        frame.render_widget(header, chunks[0]);

        let list_block = Block::default().borders(Borders::ALL).title("Questions");
        self.selection_area = Some(chunks[1]);
        if self.answers.is_empty() {
            let empty = Paragraph::new("No RolloutAnswerRaw entries found in the given file")
                .block(list_block);
            frame.render_widget(empty, chunks[1]);
        } else {
            let items: Vec<ListItem> = self
                .answers
                .iter()
                .map(|answer| {
                    let truncated_question: String = answer
                        .question
                        .lines()
                        .next()
                        .unwrap_or("")
                        .chars()
                        .take(120)
                        .collect();
                    let correctness_prefix = match &self.correctness_by_id {
                        Some(map) => {
                            let correct = map.get(&answer.id).copied().unwrap_or(false);
                            if correct { "✓ " } else { "✗ " }
                        }
                        None => "",
                    };
                    let display = format!(
                        "{}{}: {}",
                        correctness_prefix, answer.id, truncated_question
                    );
                    ListItem::new(display)
                })
                .collect();
            frame.render_stateful_widget(
                List::new(items)
                    .block(list_block)
                    .highlight_style(
                        Style::default()
                            .fg(Color::Yellow)
                            .add_modifier(Modifier::BOLD),
                    )
                    .highlight_symbol("➤ "),
                chunks[1],
                &mut self.selection_state,
            );
        }

        let footer =
            Paragraph::new("Use ↑/↓ to change selection, Enter to open a session, q to quit")
                .block(Block::default().borders(Borders::ALL).title("Controls"));
        frame.render_widget(footer, chunks[2]);
    }

    fn draw_session(&mut self, frame: &mut ratatui::Frame<'_>) {
        let view = self.browsing_view.as_ref().unwrap();
        self.selection_area = None;
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(8),
                Constraint::Min(6),
                Constraint::Length(3),
            ])
            .split(frame.area());

        let header_text = format!(
            "Question {}: {}\nModel answer: {}\nCorrect answer: {}",
            view.answer.id,
            view.answer.question,
            view.model_answer,
            view.answer.correct_answer,
        );
        let header = Paragraph::new(header_text)
            .wrap(Wrap { trim: false })
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title("Session header"),
            );
        frame.render_widget(header, chunks[0]);

        let body_chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Ratio(1, 5), Constraint::Ratio(4, 5)])
            .split(chunks[1]);

        self.log_area = Some(body_chunks[0]);
        let log_title = format!(
            "Trajectory actions ({}/{}){}",
            view.current_pos,
            view.total_ops(),
            if self.focus == PaneFocus::Log {
                " [focused]"
            } else {
                ""
            }
        );
        let log_block = Block::default()
            .borders(Borders::ALL)
            .title(log_title)
            .border_style(if self.focus == PaneFocus::Log {
                Style::default()
                    .fg(Color::LightGreen)
                    .add_modifier(Modifier::BOLD)
            } else {
                Style::default()
            });
        let log_items: Vec<ListItem> = view
            .operations
            .iter()
            .enumerate()
            .map(|(i, op)| ListItem::new(format!("[{i}] {}", op.to_concise_string())))
            .collect();
        let mut log_state = ListState::default();
        if view.current_pos > 0 {
            log_state.select(Some(view.current_pos - 1));
        }
        frame.render_stateful_widget(
            List::new(log_items)
                .block(log_block)
                .highlight_style(
                    Style::default()
                        .fg(Color::Cyan)
                        .add_modifier(Modifier::BOLD),
                )
                .highlight_symbol("▶ "),
            body_chunks[0],
            &mut log_state,
        );

        let prompt_text = view
            .current_prompt_display()
            .unwrap_or_else(|| "No prompt available".into());
        let action_text = view
            .current_operation_display()
            .unwrap_or_else(|| "No action yet".to_string());

        let right_chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
            .split(body_chunks[1]);

        self.prompt_area = Some(right_chunks[0]);
        self.action_area = Some(right_chunks[1]);

        let turn_info = format!(
            "display turn: {}, actual turn: {}",
            view.total_display_turns, view.total_actual_turns
        );
        let prompt_block = Block::default()
            .borders(Borders::ALL)
            .title(format!(
                "Prompt (role: {}) [{}]{}",
                "Prompt Label",
                turn_info,
                if self.focus == PaneFocus::Prompt {
                    " [focused]"
                } else {
                    ""
                }
            ))
            .border_style(if self.focus == PaneFocus::Prompt {
                Style::default()
                    .fg(Color::LightGreen)
                    .add_modifier(Modifier::BOLD)
            } else {
                Style::default()
            });
        let action_block = Block::default()
            .borders(Borders::ALL)
            .title(if self.focus == PaneFocus::Action {
                "Action [focused]"
            } else {
                "Action"
            })
            .border_style(if self.focus == PaneFocus::Action {
                Style::default()
                    .fg(Color::LightGreen)
                    .add_modifier(Modifier::BOLD)
            } else {
                Style::default()
            });

        let prompt_text_area = prompt_block.inner(right_chunks[0]);
        let action_text_area = action_block.inner(right_chunks[1]);
        let prompt_height = prompt_text_area.height as usize;
        let action_height = action_text_area.height as usize;
        let prompt_lines =
            compute_wrapped_line_count(&prompt_text, prompt_text_area, &mut self.prompt_metrics);
        let action_lines =
            compute_wrapped_line_count(&action_text, action_text_area, &mut self.action_metrics);
        let prompt_max_scroll = bottom_scroll_limit(prompt_lines, prompt_height.max(1));
        let action_max_scroll = bottom_scroll_limit(action_lines, action_height.max(1));
        self.prompt_scroll = self.prompt_scroll.min(prompt_max_scroll);
        self.action_scroll = self.action_scroll.min(action_max_scroll);

        let prompt_paragraph = Paragraph::new(prompt_text)
            .block(prompt_block)
            .wrap(Wrap { trim: false })
            .scroll((clamp_scroll(self.prompt_scroll), 0));
        let action_paragraph = Paragraph::new(action_text)
            .block(action_block)
            .wrap(Wrap { trim: false })
            .scroll((clamp_scroll(self.action_scroll), 0));
        frame.render_widget(prompt_paragraph, right_chunks[0]);
        frame.render_widget(action_paragraph, right_chunks[1]);

        let footer = Paragraph::new(
            "Left/Right/Tab: switch focus among panes; Up/Down: move action selection in log pane or scroll text panes; PgUp/PgDn/Home/End: jump action selection; Esc/q: back to tree page",
        )
        .block(Block::default().borders(Borders::ALL).title("Controls"));
        frame.render_widget(footer, chunks[2]);
    }
}

fn adjust_scroll_offset(offset: &mut usize, delta: isize) {
    let magnitude = delta.unsigned_abs() as usize;
    if delta < 0 {
        *offset = offset.saturating_sub(magnitude);
    } else {
        *offset = offset.saturating_add(magnitude);
    }
}

fn contains_point(rect: Rect, column: u16, row: u16) -> bool {
    column >= rect.x && column < rect.x + rect.width && row >= rect.y && row < rect.y + rect.height
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
                .map_or(false, |cell| !cell.symbol().trim().is_empty())
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

struct SessionView {
    answer: RolloutTrajectory,
    model_answer: String,
    operations: Vec<RolloutAction>,
    current_pos: usize,
    total_display_turns: usize,
    total_actual_turns: usize,
}

struct TreeView {
    answer: RolloutTrajectory,
    tree_lines: Vec<String>,
    tree_line_node_ids: Vec<usize>,
    selected_tree_line_index: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NodeAbbreviation {
    Voff,
    Von,
    Vow,
    Voc,
}

impl NodeAbbreviation {
    fn as_str(self) -> &'static str {
        match self {
            NodeAbbreviation::Voff => "VOF",
            NodeAbbreviation::Von => "VON",
            NodeAbbreviation::Vow => "VOW",
            NodeAbbreviation::Voc => "VOC",
        }
    }
}

fn compact_node_label(tree: &Tree, node_id: usize) -> String {
    let node = &tree.nodes[node_id];
    let abbreviation = derive_node_abbreviation_from_actions(&node.step.action_log);
    format!("{:02}{}", node.node_id % 100, abbreviation.as_str())
}

fn validate_tree_for_browser(tree: &Tree) {
    for i in 0..tree.nodes.len() {
        let node = &tree.nodes[i];
        assert_eq!(
            node.node_id, i,
            "Tree invariant violated: node index must equal node_id"
        );
        if i == 0 {
            assert!(
                node.parent_id.is_none(),
                "Root node must not have parent_id"
            );
        } else {
            let parent_id = node
                .parent_id
                .expect("Non-root node must have parent_id for browser traversal");
            assert!(
                parent_id < tree.nodes.len(),
                "Non-root node parent_id must point to an existing node"
            );
        }
    }
    for parent in &tree.nodes {
        if let Some(child_id) = parent.verifier_on_child_id {
            assert!(
                child_id < tree.nodes.len(),
                "verifier_on_child_id must point to existing node"
            );
        }
        if let Some(child_id) = parent.verifier_off_child_id {
            assert!(
                child_id < tree.nodes.len(),
                "verifier_off_child_id must point to existing node"
            );
        }
    }
}

fn collect_root_to_node_action_sequence(tree: &Tree, selected_node_id: usize) -> Vec<RolloutAction> {
    assert!(
        selected_node_id < tree.nodes.len(),
        "Selected node id must exist in tree"
    );
    let mut node_ids_from_selected_to_root: Vec<usize> = Vec::new();
    let mut cursor = Some(selected_node_id);
    while let Some(node_id) = cursor {
        let node = &tree.nodes[node_id];
        assert_eq!(
            node.node_id, node_id,
            "Tree traversal requires node index to equal node_id"
        );
        node_ids_from_selected_to_root.push(node_id);
        cursor = node.parent_id;
    }
    node_ids_from_selected_to_root.reverse();

    let mut actions: Vec<RolloutAction> = Vec::new();
    for node_id in node_ids_from_selected_to_root {
        let node = &tree.nodes[node_id];
        actions.extend(node.step.action_log.iter().cloned());
    }
    actions
}

fn validate_session_log_for_prompt_replay(
    question: &str,
    source_tree: &Tree,
    operations: &[RolloutAction],
) {
    assert!(
        !operations.is_empty(),
        "Selected root-to-node trajectory must contain at least one action"
    );
    assert!(
        matches!(operations[0], RolloutAction::VerifierComment(_)),
        "Selected trajectory must start with VerifierComment"
    );
    for prefix_len in 1..=operations.len() {
        let prefix_log = TrajectoryActionLog(operations[..prefix_len].to_vec());
        let state = TrajectoryState::from_session_log(question.to_string(), prefix_log, source_tree);
        let _ = get_prompt_according_to_session_status(&state);
    }
}

fn derive_node_abbreviation_from_actions(action_log: &[RolloutAction]) -> NodeAbbreviation {
    let mut verifier_comment: Option<Option<credit_assignment::multi_agent::session::VerifierComment>> =
        None;
    let mut planner_decision: Option<NextStepDecision> = None;
    let mut has_intervention = false;

    for action in action_log {
        match action {
            RolloutAction::VerifierComment(comment) => {
                assert!(
                    verifier_comment.is_none(),
                    "Each node action log must contain exactly one VerifierComment"
                );
                verifier_comment = Some(comment.clone());
            }
            RolloutAction::PlannerDecideNextStep(mode) => {
                assert!(
                    planner_decision.is_none(),
                    "Each node action log must contain at most one PlannerDecideNextStep"
                );
                planner_decision = Some(mode.clone());
            }
            RolloutAction::ToolCallResponse(credit_assignment::multi_agent::session::ToolResponse::Intervention(_)) => {
                has_intervention = true;
            }
            _ => {}
        }
    }

    let verifier_comment = verifier_comment
        .expect("Each node action log must contain exactly one VerifierComment for abbreviation");

    if planner_decision.is_none() {
        assert!(
            !action_log.is_empty(),
            "Node action log must not be empty when deriving abbreviation"
        );
        assert!(
            has_intervention
                || action_log
                    .iter()
                    .any(|action| matches!(action, RolloutAction::PlannerMakeOrChangePlan(_))),
            "Missing PlannerDecideNextStep requires intervention or downstream planner actions"
        );
        return match verifier_comment {
            None => NodeAbbreviation::Voff,
            Some(_) => NodeAbbreviation::Von,
        };
    }
    let planner_decision = planner_decision
        .expect("Planner decision must exist after non-intervention path validation");

    match (verifier_comment, planner_decision) {
        (None, NextStepDecision::Continue) => NodeAbbreviation::Voff,
        (None, NextStepDecision::OverwriteLastStep(_)) => {
            panic!("Verifier-off node cannot choose OverwriteLastStep")
        }
        (None, NextStepDecision::ChangePlan(_)) => {
            panic!("Verifier-off node cannot choose ChangePlan")
        }
        (Some(_), NextStepDecision::Continue) => NodeAbbreviation::Von,
        (Some(_), NextStepDecision::OverwriteLastStep(_)) => NodeAbbreviation::Vow,
        (Some(_), NextStepDecision::ChangePlan(_)) => NodeAbbreviation::Voc,
    }
}

fn collect_path_ids_from_leaf_to_root(tree: &Tree, leaf_node_id: usize) -> Vec<usize> {
    assert!(
        leaf_node_id < tree.nodes.len(),
        "Leaf node id must exist in tree"
    );
    let mut path_ids: Vec<usize> = Vec::new();
    let mut cursor = Some(leaf_node_id);
    while let Some(node_id) = cursor {
        let node = &tree.nodes[node_id];
        path_ids.push(node_id);
        cursor = node.parent_id;
    }
    path_ids
}

fn collect_path_ids_from_root_to_leaf(tree: &Tree, leaf_node_id: usize) -> Vec<usize> {
    let mut path_ids = collect_path_ids_from_leaf_to_root(tree, leaf_node_id);
    path_ids.reverse();
    path_ids
}

fn is_descendant_of(tree: &Tree, descendant_node_id: usize, ancestor_node_id: usize) -> bool {
    assert!(descendant_node_id < tree.nodes.len(), "descendant node id must exist");
    assert!(ancestor_node_id < tree.nodes.len(), "ancestor node id must exist");
    let mut cursor = Some(descendant_node_id);
    while let Some(node_id) = cursor {
        if node_id == ancestor_node_id {
            return true;
        }
        cursor = tree.nodes[node_id].parent_id;
    }
    false
}

fn pick_first_leaf_in_subtree(tree: &Tree, subtree_root_node_id: usize) -> usize {
    for &leaf_node_id in &tree.leaf_node_ids {
        if is_descendant_of(tree, leaf_node_id, subtree_root_node_id) {
            return leaf_node_id;
        }
    }
    panic!("No leaf found in subtree rooted at node {subtree_root_node_id}");
}

fn collect_leaf_order_by_focus(tree: &Tree) -> Vec<usize> {
    assert!(!tree.leaf_node_ids.is_empty(), "Tree must contain at least one leaf");
    let total_leaves = tree.leaf_node_ids.len();
    let mut ordered_leaves: Vec<usize> = Vec::new();
    let mut seen_leaf = vec![false; tree.nodes.len()];
    let mut visited_sibling_child_nodes: HashSet<usize> = HashSet::new();
    let mut stack: Vec<usize> = vec![tree.leaf_node_ids[0]];

    while let Some(focus_leaf_node_id) = stack.pop() {
        assert!(focus_leaf_node_id < tree.nodes.len(), "focus leaf must exist");
        if seen_leaf[focus_leaf_node_id] {
            continue;
        }
        seen_leaf[focus_leaf_node_id] = true;
        ordered_leaves.push(focus_leaf_node_id);

        let path_root_to_leaf = collect_path_ids_from_root_to_leaf(tree, focus_leaf_node_id);
        assert!(!path_root_to_leaf.is_empty(), "root-to-leaf path must be non-empty");

        let mut sibling_leaf_candidates: Vec<usize> = Vec::new();
        for depth in (0..path_root_to_leaf.len().saturating_sub(1)).rev() {
            let parent_node_id = path_root_to_leaf[depth];
            let focus_child_node_id = path_root_to_leaf[depth + 1];
            let parent = &tree.nodes[parent_node_id];
            let on_child = parent.verifier_on_child_id;
            let off_child = parent.verifier_off_child_id;
            let has_two_children = on_child.is_some() && off_child.is_some();
            if !has_two_children {
                continue;
            }
            let sibling_child_node_id = if on_child == Some(focus_child_node_id) {
                off_child.expect("off child must exist when parent has two children")
            } else if off_child == Some(focus_child_node_id) {
                on_child.expect("on child must exist when parent has two children")
            } else {
                panic!("focus path child must be one of the parent children");
            };

            if !visited_sibling_child_nodes.insert(sibling_child_node_id) {
                continue;
            }

            let sibling_focus_leaf = pick_first_leaf_in_subtree(tree, sibling_child_node_id);
            sibling_leaf_candidates.push(sibling_focus_leaf);
        }

        for sibling_leaf in sibling_leaf_candidates.into_iter().rev() {
            if !seen_leaf[sibling_leaf] {
                stack.push(sibling_leaf);
            }
        }
    }

    assert_eq!(
        ordered_leaves.len(),
        total_leaves,
        "Focused iterative leaf ordering must cover all leaves"
    );
    ordered_leaves
}

fn node_depth_from_root(tree: &Tree, node_id: usize) -> usize {
    assert!(node_id < tree.nodes.len(), "node id must exist");
    let mut depth = 0usize;
    let mut cursor = tree.nodes[node_id].parent_id;
    while let Some(parent_id) = cursor {
        depth += 1;
        cursor = tree.nodes[parent_id].parent_id;
    }
    depth
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

fn build_tree_lines(tree: &Tree) -> (Vec<usize>, Vec<String>) {
    let root_id = tree
        .root_node_id
        .expect("Tree browser requires root_node_id to exist");
    assert_eq!(root_id, 0, "Tree browser expects root node id to be 0");
    let ordered_leaf_node_ids = collect_leaf_order_by_focus(tree);
    let line_count = ordered_leaf_node_ids.len();
    let mut canvas: Vec<Vec<char>> = vec![Vec::new(); line_count];

    let mut node_row: Vec<Option<usize>> = vec![None; tree.nodes.len()];
    for (row, &leaf_node_id) in ordered_leaf_node_ids.iter().enumerate() {
        let path_root_to_leaf = collect_path_ids_from_root_to_leaf(tree, leaf_node_id);
        assert_eq!(path_root_to_leaf[0], root_id, "Every path must start from root");
        for &node_id in &path_root_to_leaf {
            if node_row[node_id].is_none() {
                node_row[node_id] = Some(row);
            }
        }
    }

    let mut node_col: Vec<usize> = vec![0; tree.nodes.len()];
    for node_id in 0..tree.nodes.len() {
        let depth = node_depth_from_root(tree, node_id);
        node_col[node_id] = depth * 8;
    }

    for node_id in 0..tree.nodes.len() {
        let Some(row) = node_row[node_id] else {
            continue;
        };
        let col = node_col[node_id];
        let label = compact_node_label(tree, node_id);
        assert_eq!(label.chars().count(), 5, "Node label must have exactly 5 chars");
        write_pattern(&mut canvas, row, col, &label);
    }

    for parent_id in 0..tree.nodes.len() {
        let Some(parent_row) = node_row[parent_id] else {
            continue;
        };
        let parent_col = node_col[parent_id];
        let edge_col = parent_col + 5;
        let parent = &tree.nodes[parent_id];
        let children: [Option<usize>; 2] = [parent.verifier_on_child_id, parent.verifier_off_child_id];
        for child_node_id in children.into_iter().flatten() {
            let child_row = node_row[child_node_id]
                .expect("Child node in rendered tree must have a resolved row");
            if child_row == parent_row {
                let has_lower_sibling = children
                    .into_iter()
                    .flatten()
                    .filter(|id| *id != child_node_id)
                    .any(|other_id| {
                        let other_row = node_row[other_id]
                            .expect("Sibling node in rendered tree must have a resolved row");
                        other_row > parent_row
                    });
                if has_lower_sibling {
                    write_pattern(&mut canvas, parent_row, edge_col, "━┳━");
                } else {
                    write_pattern(&mut canvas, parent_row, edge_col, "━━━");
                }
            } else {
                assert!(
                    child_row > parent_row,
                    "Child row must be same row or below parent row"
                );
                for row in (parent_row + 1)..child_row {
                    write_pattern(&mut canvas, row, edge_col, " ┃ ");
                }
                write_pattern(&mut canvas, child_row, edge_col, " ┗━");
            }
        }
    }

    for &leaf_node_id in &ordered_leaf_node_ids {
        let row = node_row[leaf_node_id].expect("Leaf node must have a rendered row");
        let col = node_col[leaf_node_id] + 5;
        let judgment = tree
            .leaf_node_judgments
            .get(&leaf_node_id)
            .expect("Rendered leaf must have correctness judgment");
        let suffix = if judgment.is_correct { "━━━✓" } else { "━━━✗" };
        write_pattern(&mut canvas, row, col, suffix);
    }

    let mut lines: Vec<String> = Vec::with_capacity(canvas.len());
    for row in canvas {
        let mut line: String = row.into_iter().collect();
        line = line.trim_end().to_string();
        lines.push(line);
    }
    assert_eq!(
        lines.len(),
        ordered_leaf_node_ids.len(),
        "Tree line count must match leaf node count"
    );
    (ordered_leaf_node_ids, lines)
}

fn slice_line_from_char_offset(line: &str, offset: usize) -> String {
    line.chars().skip(offset).collect::<String>()
}

fn count_display_turns(operations: &[RolloutAction]) -> usize {
    operations
        .iter()
        .filter(|action| matches!(action, RolloutAction::PlannerEndStep))
        .count()
}

impl TreeView {
    fn new(answer: RolloutTrajectory) -> Self {
        validate_tree_for_browser(&answer.trajectory);
        let (tree_line_node_ids, tree_lines) = build_tree_lines(&answer.trajectory);
        let current_node_id = answer
            .trajectory
            .current_node_id
            .expect("Browser requires current_node_id to reconstruct selected path");
        assert!(
            answer.trajectory.nodes[current_node_id]
                .verifier_on_child_id
                .is_none()
                && answer.trajectory.nodes[current_node_id]
                    .verifier_off_child_id
                    .is_none(),
            "Current node must be a leaf for leaf-aligned tree view"
        );
        let selected_tree_line_index = tree_line_node_ids
            .iter()
            .position(|node_id| *node_id == current_node_id)
            .expect("Current node id must appear in rendered leaf lines");
        Self {
            answer,
            tree_lines,
            tree_line_node_ids,
            selected_tree_line_index,
        }
    }

    fn selected_node_id(&self) -> usize {
        self.tree_line_node_ids[self.selected_tree_line_index]
    }

    fn move_tree_selection_by(&mut self, delta: isize) {
        let total = self.tree_lines.len();
        if total == 0 {
            return;
        }
        let next = self.selected_tree_line_index as isize + delta;
        self.selected_tree_line_index = next.clamp(0, total as isize - 1) as usize;
    }

    fn move_tree_selection_to_start(&mut self) {
        assert!(!self.tree_lines.is_empty(), "Tree lines must not be empty");
        self.selected_tree_line_index = 0;
    }

    fn move_tree_selection_to_end(&mut self) {
        assert!(!self.tree_lines.is_empty(), "Tree lines must not be empty");
        self.selected_tree_line_index = self.tree_lines.len() - 1;
    }

    fn select_tree_line_by_index(&mut self, index: usize) {
        assert!(index < self.tree_lines.len(), "Tree line index must be in range");
        self.selected_tree_line_index = index;
    }
}

impl SessionView {
    fn new(answer: RolloutTrajectory, selected_node_id: usize) -> Self {
        validate_tree_for_browser(&answer.trajectory);
        let operations = collect_root_to_node_action_sequence(&answer.trajectory, selected_node_id);
        validate_session_log_for_prompt_replay(&answer.question, &answer.trajectory, &operations);
        let model_answer = answer
            .trajectory
            .leaf_node_judgments
            .get(&selected_node_id)
            .expect("SessionView requires selected leaf to have correctness judgment")
            .model_answer
            .clone();
        let total_display_turns = count_display_turns(&operations);
        let total_actual_turns = total_display_turns;
        Self {
            answer,
            model_answer,
            operations,
            current_pos: 0,
            total_display_turns,
            total_actual_turns,
        }
    }

    fn total_ops(&self) -> usize {
        self.operations.len()
    }

    fn move_by(&mut self, delta: isize) {
        if self.operations.is_empty() {
            self.current_pos = 0;
            return;
        }
        let total = self.total_ops();
        let next = self.current_pos as isize + delta;
        self.current_pos = next.clamp(0, total as isize) as usize;
    }

    fn move_to_start(&mut self) {
        self.current_pos = 0;
    }

    fn move_to_end(&mut self) {
        self.current_pos = self.total_ops();
    }

    fn current_operation_display(&self) -> Option<String> {
        if self.current_pos == 0 {
            None
        } else {
            let operation = self.operations.get(self.current_pos - 1)?;
            Some(operation.to_pretty_string())
        }
    }

    fn current_prompt_display(&self) -> Option<String> {
        if self.operations.is_empty() {
            return None;
        }
        let prefix_len = self.current_pos.saturating_sub(1);
        if prefix_len == 0 {
            return Some(
                "Prompt is undefined when the aligned replay prefix is empty (before the first valid prompt-bearing state). Scroll further to view aligned prompt reconstruction.".to_string(),
            );
        }
        let prefix_log = TrajectoryActionLog(
            self.operations
                .iter()
                .take(prefix_len)
                .cloned()
                .collect(),
        );
        let state: TrajectoryState<'_> = TrajectoryState::from_session_log(
            self.answer.question.clone(),
            prefix_log,
            &self.answer.trajectory,
        );
        let (prompt_before_assistant, prompt_after_assistant) =
            get_prompt_according_to_session_status(&state);
        Some(format!(
            "{}\n==========Assistant==========\n{}",
            prompt_before_assistant, prompt_after_assistant
        ))
    }
}
