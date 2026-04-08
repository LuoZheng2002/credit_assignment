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

use credit_assignment::multi_agent::generate_rollout_answers::RolloutAnswerRaw;
use credit_assignment::multi_agent::rollout::{get_planner_prompt, get_verifier_prompt};
use credit_assignment::multi_agent::session::{
    ModelOperation, PlannerStatus, SessionState, SessionStatus,
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
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();
    let answers = load_rollout_answers(&args.file)?;
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;
    let result = run_app(&mut terminal, answers);
    disable_raw_mode()?;
    execute!(
        terminal.backend_mut(),
        LeaveAlternateScreen,
        DisableMouseCapture
    )?;
    terminal.show_cursor()?;
    result
}

fn load_rollout_answers(path: &PathBuf) -> Result<Vec<RolloutAnswerRaw>, Box<dyn Error>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut answers = Vec::new();
    for line in reader.lines() {
        let line = line?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let answer: RolloutAnswerRaw = serde_json::from_str(trimmed)?;
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
    answers: Vec<RolloutAnswerRaw>,
) -> Result<(), Box<dyn Error>> {
    let mut app = App::new(answers);
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
    Quit,
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
    answers: Vec<RolloutAnswerRaw>,
    selection_state: ListState,
    browsing_view: Option<SessionView>,
    focus: PaneFocus,
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
    fn new(answers: Vec<RolloutAnswerRaw>) -> Self {
        let mut selection_state = ListState::default();
        if !answers.is_empty() {
            selection_state.select(Some(0));
        }
        Self {
            answers,
            selection_state,
            browsing_view: None,
            focus: PaneFocus::Log,
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
                    false
                }
                BrowsingAction::Quit => true,
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
                    self.step_selection(-1);
                    false
                }
                MouseEventKind::ScrollDown => {
                    self.step_selection(1);
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

    fn step_selection(&mut self, delta: isize) {
        if self.answers.is_empty() {
            return;
        }
        let current = self.selection_state.selected().unwrap_or(0);
        let next = if delta < 0 {
            current.saturating_sub(delta.unsigned_abs() as usize)
        } else {
            current.saturating_add(delta as usize)
        };
        let clamped = next.min(self.answers.len() - 1);
        self.selection_state.select(Some(clamped));
    }

    fn open_selected_question(&mut self) {
        if let Some(selected) = self.selection_state.selected() {
            if selected < self.answers.len() {
                let answer = self.answers[selected].clone();
                self.browsing_view = Some(SessionView::new(answer));
                self.focus = PaneFocus::Log;
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
                    view.move_to_start();
                    BrowsingAction::Continue
                }
                KeyCode::End => {
                    view.move_to_end();
                    BrowsingAction::Continue
                }
                KeyCode::PageUp => {
                    view.move_by(-10);
                    BrowsingAction::Continue
                }
                KeyCode::PageDown => {
                    view.move_by(10);
                    BrowsingAction::Continue
                }
                KeyCode::Esc => BrowsingAction::GoBack,
                KeyCode::Char('q') => BrowsingAction::Quit,
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
                    let display = format!("{}: {}", answer.id, truncated_question);
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
                Constraint::Length(4),
                Constraint::Min(6),
                Constraint::Length(3),
            ])
            .split(frame.area());

        let header_text = format!(
            "Question {}: {}",
            view.answer.id,
            view.answer
                .question
                .lines()
                .next()
                .unwrap_or("<empty question>")
        );
        let header = Paragraph::new(header_text).block(
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
            "Session log progress ({}/{}){}",
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
        let visible_ops = view.operations.iter().enumerate().take(view.current_pos);
        let log_items: Vec<ListItem> = visible_ops
            .map(|(index, _)| {
                ListItem::new(format!(
                    "[{index}] {role}",
                    index = index,
                    role = view.operation_roles[index].label(),
                ))
            })
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

        let (prompt_label, prompt_text) = view
            .current_prompt()
            .map(|(role, text)| (role.label().to_string(), text))
            .unwrap_or_else(|| ("Planner".into(), "No prompt available".into()));
        let action_text = view
            .current_operation()
            .map(|entry| entry.to_pretty_string())
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
                prompt_label,
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
            .wrap(Wrap { trim: true })
            .scroll((clamp_scroll(self.prompt_scroll), 0));
        let action_paragraph = Paragraph::new(action_text)
            .block(action_block)
            .wrap(Wrap { trim: true })
            .scroll((clamp_scroll(self.action_scroll), 0));
        frame.render_widget(prompt_paragraph, right_chunks[0]);
        frame.render_widget(action_paragraph, right_chunks[1]);

        let footer = Paragraph::new(
            "Left/Right: switch focus between log, prompt, and action panes; Up/Down: scroll the focused pane (or move the log when it is focused); PgUp/PgDn: jump log positions; Esc: go back to selection; q: quit",
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
        .wrap(Wrap { trim: true })
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
    answer: RolloutAnswerRaw,
    operations: Vec<ModelOperation>,
    operation_roles: Vec<PromptRole>,
    current_pos: usize,
    total_display_turns: usize,
    total_actual_turns: usize,
}

impl SessionView {
    fn new(answer: RolloutAnswerRaw) -> Self {
        let operations = answer.trajectory.operations().to_vec();
        let operation_roles = compute_operation_roles(&operations);
        let mut final_state = SessionState::new();
        for operation in operations.iter() {
            final_state.update(operation.clone());
        }
        let total_display_turns = final_state.total_display_rounds();
        let total_actual_turns = answer.trajectory.total_actual_rounds();
        Self {
            answer,
            operations,
            operation_roles,
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

    fn current_operation(&self) -> Option<&ModelOperation> {
        if self.current_pos == 0 {
            None
        } else {
            self.operations.get(self.current_pos - 1)
        }
    }

    fn current_prompt(&self) -> Option<(PromptRole, String)> {
        if self.operations.is_empty() {
            return None;
        }
        let state = self.session_state_at(self.current_pos.saturating_sub(1));
        let (history_prev_steps, history_curr_step) = match state.session_status {
            SessionStatus::PlannerTurn => (
                state.to_history_prev_steps(true),
                state.to_history_curr_step(true),
            ),
            SessionStatus::VerifierTurn => (
                state.to_history_prev_steps(false),
                state.to_history_curr_step(false),
            ),
        };
        let prompt = match state.session_status {
            SessionStatus::PlannerTurn => {
                let prompt = get_planner_prompt(
                    &self.answer.question,
                    state.planner_status,
                    &history_prev_steps,
                    &history_curr_step,
                );
                let role = if matches!(state.planner_status, PlannerStatus::PlannerChoosingMode) {
                    PromptRole::PlannerChoice
                } else {
                    PromptRole::PlannerStep
                };
                Some((role, prompt))
            }
            SessionStatus::VerifierTurn => Some((
                PromptRole::Verifier,
                get_verifier_prompt(
                    &self.answer.question,
                    &history_prev_steps,
                    &history_curr_step,
                ),
            )),
        };
        prompt
    }

    fn session_state_at(&self, upto: usize) -> SessionState {
        let mut state = SessionState::new();
        for operation in self.operations.iter().take(upto) {
            state.update(operation.clone());
        }
        state
    }
}

#[derive(Copy, Clone)]
enum PromptRole {
    PlannerChoice,
    PlannerStep,
    Verifier,
}

impl PromptRole {
    fn label(self) -> &'static str {
        match self {
            PromptRole::PlannerChoice => "Planner (choose mode)",
            PromptRole::PlannerStep => "Planner (step reasoning)",
            PromptRole::Verifier => "Verifier",
        }
    }
}

fn compute_operation_roles(operations: &[ModelOperation]) -> Vec<PromptRole> {
    let mut state = SessionState::new();
    let mut roles = Vec::with_capacity(operations.len());
    for operation in operations {
        let role = match state.session_status {
            SessionStatus::PlannerTurn => match state.planner_status {
                PlannerStatus::PlannerChoosingMode => PromptRole::PlannerChoice,
                _ => PromptRole::PlannerStep,
            },
            SessionStatus::VerifierTurn => PromptRole::Verifier,
        };
        roles.push(role);
        state.update(operation.clone());
    }
    roles
}
