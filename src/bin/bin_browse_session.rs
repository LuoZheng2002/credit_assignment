use std::collections::HashMap;
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
use credit_assignment::multi_agent::session::{NextStepDecision, RolloutAction, Tree};

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
    browsing_view: Option<SessionView>,
    focus: PaneFocus,
    tree_horizontal_scroll: usize,
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
            browsing_view: None,
            focus: PaneFocus::Log,
            tree_horizontal_scroll: 0,
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
        } else {
            self.draw_selection(frame);
        }
    }

    fn handle_key(&mut self, key: KeyEvent) -> bool {
        if self.browsing_view.is_some() {
            match self.handle_browsing_key(key) {
                BrowsingAction::Continue => false,
                BrowsingAction::GoBack => {
                    self.browsing_view = None;
                    self.tree_horizontal_scroll = 0;
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
                    if self.focus == PaneFocus::Log {
                        if let Some(local_index) =
                            self.tree_line_index_from_mouse(mouse_event.column, mouse_event.row)
                        {
                            if let Some(view) = self.browsing_view.as_mut() {
                                if local_index < view.tree_lines.len() {
                                    view.select_tree_line_by_index(local_index);
                                }
                            }
                        }
                    }
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
                self.browsing_view = Some(SessionView::new(answer));
                self.focus = PaneFocus::Log;
                self.tree_horizontal_scroll = 0;
                self.prompt_scroll = 0;
                self.action_scroll = 0;
                self.selection_area = None;
            }
        }
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

    fn handle_browsing_key(&mut self, key: KeyEvent) -> BrowsingAction {
        let mut new_focus = self.focus;
        let mut new_tree_horizontal_scroll = self.tree_horizontal_scroll;
        let mut new_prompt_scroll = self.prompt_scroll;
        let mut new_action_scroll = self.action_scroll;
        let action = {
            let view = self.browsing_view.as_mut().unwrap();
            match key.code {
                KeyCode::Left => {
                    if new_focus == PaneFocus::Log {
                        new_tree_horizontal_scroll = new_tree_horizontal_scroll.saturating_sub(1);
                    } else {
                        new_focus = new_focus.prev();
                    }
                    BrowsingAction::Continue
                }
                KeyCode::Right => {
                    if new_focus == PaneFocus::Log {
                        new_tree_horizontal_scroll = new_tree_horizontal_scroll.saturating_add(1);
                    } else {
                        new_focus = new_focus.next();
                    }
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
                        PaneFocus::Log => view.move_tree_selection_by(-1),
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
                        PaneFocus::Log => view.move_tree_selection_by(1),
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
                    if new_focus == PaneFocus::Log {
                        view.move_tree_selection_to_start();
                    }
                    BrowsingAction::Continue
                }
                KeyCode::End => {
                    if new_focus == PaneFocus::Log {
                        view.move_tree_selection_to_end();
                    }
                    BrowsingAction::Continue
                }
                KeyCode::PageUp => {
                    if new_focus == PaneFocus::Log {
                        view.move_tree_selection_by(-10);
                    }
                    BrowsingAction::Continue
                }
                KeyCode::PageDown => {
                    if new_focus == PaneFocus::Log {
                        view.move_tree_selection_by(10);
                    }
                    BrowsingAction::Continue
                }
                KeyCode::Esc | KeyCode::Char('q') => BrowsingAction::GoBack,
                _ => BrowsingAction::Continue,
            }
        };
        self.focus = new_focus;
        self.tree_horizontal_scroll = new_tree_horizontal_scroll;
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
            PaneFocus::Log => view.move_tree_selection_by(delta),
            PaneFocus::Prompt => adjust_scroll_offset(&mut self.prompt_scroll, delta),
            PaneFocus::Action => adjust_scroll_offset(&mut self.action_scroll, delta),
        }
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
            view.answer.model_answer,
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
            "Tree view (selected leaf: {}, hscroll: {}){}",
            view.selected_node_id(),
            self.tree_horizontal_scroll,
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
            .tree_lines
            .iter()
            .map(|line| ListItem::new(slice_line_from_char_offset(line, self.tree_horizontal_scroll)))
            .collect();
        let mut log_state = ListState::default();
        log_state.select(Some(view.selected_tree_line_index));
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
            "Tab/Shift+Tab or Left/Right on non-tree pane: switch focus; Left/Right on tree pane: horizontal scroll; Up/Down on tree pane: move selected leaf; PgUp/PgDn/Home/End: jump leaf selection; Esc: go back; q: quit",
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
    tree_lines: Vec<String>,
    tree_line_node_ids: Vec<usize>,
    selected_tree_line_index: usize,
    operations: Vec<RolloutAction>,
    total_display_turns: usize,
    total_actual_turns: usize,
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
            NodeAbbreviation::Voff => "VOFF",
            NodeAbbreviation::Von => "VON",
            NodeAbbreviation::Vow => "VOW",
            NodeAbbreviation::Voc => "VOC",
        }
    }
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

fn derive_node_abbreviation_from_actions(action_log: &[RolloutAction]) -> NodeAbbreviation {
    let mut verifier_comment: Option<Option<credit_assignment::multi_agent::session::VerifierComment>> =
        None;
    let mut planner_decision: Option<NextStepDecision> = None;

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
            _ => {}
        }
    }

    let verifier_comment = verifier_comment
        .expect("Each node action log must contain exactly one VerifierComment for abbreviation");
    let planner_decision = planner_decision
        .expect("Each node action log must contain one PlannerDecideNextStep for abbreviation");

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
    path_ids.reverse();
    path_ids
}

fn node_label(tree: &Tree, node_id: usize) -> String {
    let node = &tree.nodes[node_id];
    let abbreviation = derive_node_abbreviation_from_actions(&node.step.action_log);
    format!("{} {}", node.node_id, abbreviation.as_str())
}

fn build_tree_lines(tree: &Tree) -> (Vec<usize>, Vec<String>) {
    let root_id = tree
        .root_node_id
        .expect("Tree browser requires root_node_id to exist");
    assert_eq!(root_id, 0, "Tree browser expects root node id to be 0");

    let mut leaf_node_ids: Vec<usize> = Vec::new();
    for node in &tree.nodes {
        if node.verifier_on_child_id.is_none() && node.verifier_off_child_id.is_none() {
            leaf_node_ids.push(node.node_id);
        }
    }
    assert!(!leaf_node_ids.is_empty(), "Tree must contain at least one leaf");

    let mut lines: Vec<String> = Vec::new();
    for &leaf_node_id in &leaf_node_ids {
        let path_ids = collect_path_ids_from_leaf_to_root(tree, leaf_node_id);
        assert_eq!(
            path_ids[0], root_id,
            "Every leaf path must start from the root"
        );
        let mut tokens: Vec<String> = Vec::new();
        for (i, path_node_id) in path_ids.iter().enumerate() {
            if i == 0 {
                tokens.push(node_label(tree, *path_node_id));
            } else {
                tokens.push(format!("└─ {}", node_label(tree, *path_node_id)));
            }
        }
        lines.push(tokens.join(" "));
    }
    assert_eq!(
        lines.len(),
        leaf_node_ids.len(),
        "Tree line count must match leaf node count"
    );
    (leaf_node_ids, lines)
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

impl SessionView {
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
        let operations = collect_root_to_node_action_sequence(&answer.trajectory, current_node_id);
        let total_display_turns = count_display_turns(&operations);
        let total_actual_turns = total_display_turns;
        Self {
            answer,
            tree_lines,
            tree_line_node_ids,
            selected_tree_line_index,
            operations,
            total_display_turns,
            total_actual_turns,
        }
    }

    fn selected_node_id(&self) -> usize {
        self.tree_line_node_ids[self.selected_tree_line_index]
    }

    fn refresh_selected_path_actions(&mut self) {
        let node_id = self.selected_node_id();
        self.operations = collect_root_to_node_action_sequence(&self.answer.trajectory, node_id);
        self.total_display_turns = count_display_turns(&self.operations);
        self.total_actual_turns = self.total_display_turns;
    }

    fn move_tree_selection_by(&mut self, delta: isize) {
        let total = self.tree_lines.len();
        if total == 0 {
            return;
        }
        let next = self.selected_tree_line_index as isize + delta;
        self.selected_tree_line_index = next.clamp(0, total as isize - 1) as usize;
        self.refresh_selected_path_actions();
    }

    fn move_tree_selection_to_start(&mut self) {
        assert!(!self.tree_lines.is_empty(), "Tree lines must not be empty");
        self.selected_tree_line_index = 0;
        self.refresh_selected_path_actions();
    }

    fn move_tree_selection_to_end(&mut self) {
        assert!(!self.tree_lines.is_empty(), "Tree lines must not be empty");
        self.selected_tree_line_index = self.tree_lines.len() - 1;
        self.refresh_selected_path_actions();
    }

    fn select_tree_line_by_index(&mut self, index: usize) {
        assert!(index < self.tree_lines.len(), "Tree line index must be in range");
        self.selected_tree_line_index = index;
        self.refresh_selected_path_actions();
    }

    fn current_operation_display(&self) -> Option<String> {
        if self.operations.is_empty() {
            None
        } else {
            let mut lines: Vec<String> = Vec::new();
            for index in 0..self.operations.len() {
                lines.push(format!("[{index}] {}", self.operations[index].to_pretty_string()));
            }
            Some(lines.join("\n\n"))
        }
    }

    fn current_prompt_display(&self) -> Option<String> {
        Some(format!(
            "Selected node: {}\nAction path length: {}\n\nPrompt reconstruction is not part of the current browser scope.",
            self.selected_node_id(),
            self.operations.len(),
        ))
    }
}
