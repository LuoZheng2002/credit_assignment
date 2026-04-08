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
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::widgets::{Block, Borders, List, ListItem, ListState, Paragraph, Wrap};
use serde_json;

use credit_assignment::multi_agent::generate_rollout_answers::RolloutAnswerRaw;
use credit_assignment::multi_agent::rollout::{get_planner_prompt, get_verifier_prompt};
use credit_assignment::multi_agent::session::{
    ModelOperation, PlannerStatus, SessionLog, SessionState, SessionStatus,
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
        let visible_ops: Vec<&PromptedOperation> =
            view.prompted_ops.iter().take(view.current_pos).collect();
        let log_items: Vec<ListItem> = visible_ops
            .iter()
            .map(|entry| {
                ListItem::new(format!(
                    "[{index}] {role}",
                    index = entry.index,
                    role = entry.context.role_label(),
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

        let main_prompt = view.current_context();
        let prompt_label = main_prompt
            .map(|ctx| ctx.role_label().to_string())
            .unwrap_or_else(|| "Planner".into());
        let prompt_text = main_prompt
            .map(|ctx| ctx.prompt.as_str())
            .unwrap_or("No prompt available");
        let action_text = view
            .current_operation()
            .map(|entry| entry.operation.to_pretty_string())
            .unwrap_or_else(|| "No action yet".to_string());

        let right_chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
            .split(body_chunks[1]);

        self.prompt_area = Some(right_chunks[0]);
        self.action_area = Some(right_chunks[1]);

        let prompt_block = Block::default()
            .borders(Borders::ALL)
            .title(format!(
                "Prompt (role: {}){}",
                prompt_label,
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

struct SessionView {
    answer: RolloutAnswerRaw,
    prompted_ops: Vec<PromptedOperation>,
    current_pos: usize,
}

impl SessionView {
    fn new(answer: RolloutAnswerRaw) -> Self {
        let prompted_ops = build_prompted_operations(&answer.question, &answer.trajectory);
        Self {
            answer,
            prompted_ops,
            current_pos: 0,
        }
    }

    fn total_ops(&self) -> usize {
        self.prompted_ops.len()
    }

    fn move_by(&mut self, delta: isize) {
        if self.prompted_ops.is_empty() {
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

    fn current_operation(&self) -> Option<&PromptedOperation> {
        if self.current_pos == 0 {
            None
        } else {
            self.prompted_ops.get(self.current_pos - 1)
        }
    }

    fn current_context(&self) -> Option<&PromptContext> {
        if self.prompted_ops.is_empty() {
            return None;
        }
        if self.current_pos == 0 {
            return self.prompted_ops.first().map(|op| &op.context);
        }
        self.prompted_ops
            .get(self.current_pos - 1)
            .map(|op| &op.context)
    }
}

#[derive(Clone)]
struct PromptContext {
    role: PromptRole,
    prompt: String,
}

impl PromptContext {
    fn role_label(&self) -> &str {
        match self.role {
            PromptRole::PlannerChoice => "Planner (choose mode)",
            PromptRole::PlannerStep => "Planner (step reasoning)",
            PromptRole::Verifier => "Verifier",
        }
    }
}

#[derive(Copy, Clone)]
enum PromptRole {
    PlannerChoice,
    PlannerStep,
    Verifier,
}

struct PromptedOperation {
    index: usize,
    operation: ModelOperation,
    context: PromptContext,
}

fn build_prompted_operations(question: &str, session_log: &SessionLog) -> Vec<PromptedOperation> {
    let mut result = Vec::new();
    let mut state = SessionState::new();
    let operations = session_log.operations();
    let mut idx = 0;
    while idx < operations.len() {
        let (role, prompt_text): (PromptRole, String) = match state.session_status {
            SessionStatus::PlannerTurn => {
                let planner_status = state.planner_status;
                let history = state.to_history(true);
                let prompt = get_planner_prompt(question, planner_status, &history);
                let role = if matches!(planner_status, PlannerStatus::PlannerChoosingMode) {
                    PromptRole::PlannerChoice
                } else {
                    PromptRole::PlannerStep
                };
                (role, prompt)
            }
            SessionStatus::VerifierTurn => {
                let history = state.to_history(false);
                (
                    PromptRole::Verifier,
                    get_verifier_prompt(question, &history),
                )
            }
        };

        loop {
            if idx >= operations.len() {
                break;
            }
            let op = operations[idx].clone();
            result.push(PromptedOperation {
                index: idx,
                operation: op.clone(),
                context: PromptContext {
                    role,
                    prompt: prompt_text.clone(),
                },
            });
            state.update(op);
            idx += 1;
            let end_iteration = matches!(role, PromptRole::PlannerChoice | PromptRole::Verifier)
                || matches!(role, PromptRole::PlannerStep)
                    && matches!(
                        result.last().unwrap().operation,
                        ModelOperation::PlannerEndStep
                    );
            if end_iteration {
                break;
            }
        }
    }
    result
}
