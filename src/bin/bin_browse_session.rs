use std::error::Error;
use std::fs::File;
use std::io::{self, BufRead, BufReader, Stdout};
use std::path::PathBuf;

use clap::Parser;
use crossterm::event::{self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode, KeyEvent};
use crossterm::execute;
use crossterm::terminal::{
    EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
};
use ratatui::Terminal;
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Direction, Layout};
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
            _ => {}
        }
    }
    Ok(())
}

enum SelectionAction {
    Continue,
    Quit,
    OpenSession(usize),
}

enum BrowsingAction {
    Continue,
    GoBack,
    Quit,
}

struct App {
    answers: Vec<RolloutAnswerRaw>,
    selection_state: ListState,
    browsing_view: Option<SessionView>,
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
        }
    }

    fn draw(&mut self, frame: &mut ratatui::Frame<'_>) {
        if let Some(view) = &self.browsing_view {
            self.draw_session(frame, view);
        } else {
            self.draw_selection(frame);
        }
    }

    fn handle_key(&mut self, key: KeyEvent) -> bool {
        if let Some(view) = &mut self.browsing_view {
            match Self::handle_browsing_key(key, view) {
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
                SelectionAction::OpenSession(selected) => {
                    let answer = self.answers[selected].clone();
                    self.browsing_view = Some(SessionView::new(answer));
                    false
                }
            }
        }
    }

    fn handle_selection_key(&mut self, key: KeyEvent) -> SelectionAction {
        match key.code {
            KeyCode::Down => {
                if self.answers.is_empty() {
                    return SelectionAction::Continue;
                }
                let next = match self.selection_state.selected() {
                    Some(index) if index + 1 < self.answers.len() => index + 1,
                    _ => 0,
                };
                self.selection_state.select(Some(next));
                SelectionAction::Continue
            }
            KeyCode::Up => {
                if self.answers.is_empty() {
                    return SelectionAction::Continue;
                }
                let prev = match self.selection_state.selected() {
                    Some(index) if index > 0 => index - 1,
                    Some(_) => 0,
                    None => 0,
                };
                self.selection_state.select(Some(prev));
                SelectionAction::Continue
            }
            KeyCode::Enter => {
                if let Some(selected) = self.selection_state.selected() {
                    return SelectionAction::OpenSession(selected);
                }
                SelectionAction::Continue
            }
            KeyCode::Char('q') | KeyCode::Esc => SelectionAction::Quit,
            _ => SelectionAction::Continue,
        }
    }

    fn handle_browsing_key(key: KeyEvent, view: &mut SessionView) -> BrowsingAction {
        match key.code {
            KeyCode::Left => {
                view.move_by(-1);
                BrowsingAction::Continue
            }
            KeyCode::Right => {
                view.move_by(1);
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

    fn draw_session(&self, frame: &mut ratatui::Frame<'_>, view: &SessionView) {
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
            .constraints([Constraint::Percentage(40), Constraint::Percentage(60)])
            .split(chunks[1]);

        let log_title = format!(
            "Session log progress ({}/{})",
            view.current_pos,
            view.total_ops()
        );
        let log_block = Block::default().borders(Borders::ALL).title(log_title);
        let visible_ops: Vec<&PromptedOperation> =
            view.prompted_ops.iter().take(view.current_pos).collect();
        let log_items: Vec<ListItem> = visible_ops
            .iter()
            .map(|entry| {
                ListItem::new(format!(
                    "[{index}] ({role}) {action}",
                    index = entry.index,
                    role = entry.context.role_label(),
                    action = summarize_operation(&entry.operation)
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

        let prompt_block = Block::default().borders(Borders::ALL).title("Prompt");
        let action_block = Block::default().borders(Borders::ALL).title("Action");

        let main_prompt = view.current_context();
        let prompt_label = main_prompt
            .map(|ctx| ctx.role_label().to_string())
            .unwrap_or_else(|| "Planner".into());
        let prompt_text = main_prompt
            .map(|ctx| ctx.prompt.as_str())
            .unwrap_or("No prompt available");
        let prompt_paragraph = Paragraph::new(prompt_text)
            .block(prompt_block.title(format!("Prompt (role: {})", prompt_label)))
            .wrap(Wrap { trim: true });

        let action_text = view
            .current_operation()
            .map(|entry| entry.operation.to_pretty_string())
            .unwrap_or_else(|| "No action yet".to_string());
        let action_paragraph = Paragraph::new(action_text)
            .block(action_block)
            .wrap(Wrap { trim: true });

        let right_chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
            .split(body_chunks[1]);
        frame.render_widget(prompt_paragraph, right_chunks[0]);
        frame.render_widget(action_paragraph, right_chunks[1]);

        let footer =
            Paragraph::new("←/→ step, Home/End to jump, PgUp/PgDn fast, Esc to go back, q to quit")
                .block(Block::default().borders(Borders::ALL).title("Controls"));
        frame.render_widget(footer, chunks[2]);
    }
}

fn summarize_operation(operation: &ModelOperation) -> String {
    let raw = operation.to_pretty_string();
    raw.lines().next().unwrap_or(&raw).trim().to_string()
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
