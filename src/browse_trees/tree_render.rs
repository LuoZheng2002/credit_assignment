use std::collections::BTreeMap;
use std::error::Error;
use std::hash::{Hash, Hasher};
use std::io::Stdout;
use std::marker::PhantomData;
use std::path::Path;
use std::time::{Duration, Instant};

use clap::ValueEnum;

use crate::config_paths::{ConfigPaths, config_paths_file_path_from_action_logs_path};
use crate::direct_tool::hybrid_dataset::{
    DatasetSplit, DatasetSplitEnum, HybridDatasetQuestion, QuestionFlatId, Testing, Training,
    Validation, open_hybrid_dataset,
};
use crate::direct_tool::tree_action::DirectTreeAction;
use crate::judge_correctness::CorrectnessJudgment;
use crate::{
    direct_tool::{
        posterior_calculation_config::{PosteriorCalculationConfig, PosteriorHyperparameters},
        rollout_config::DirectRolloutConfig,
        tree::{ContentIndex, DirectTree, Segment, SegmentContent, SegmentId},
        tree_action_log::{DirectTreeActionLog, open_action_logs},
        tree_to_action::TokenBranchingScore,
    },
    json_toml_utils::read_json,
    llm_model::{
        Gemma3_4BIt, Llama31_8BInstruct, LlmModelMarker, LlmModelName, Mistral7BInstructV03,
        Qwen3_4B, Qwen3_06B, Qwen25_7B, Qwen35_4B, Qwen35_08B,
    },
};
use crossterm::event::{self, Event, KeyCode, KeyEvent, MouseButton, MouseEvent, MouseEventKind};
use ratatui::Terminal;
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Direction, Layout, Position, Rect};
use ratatui::prelude::Widget;
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span, Text};
use ratatui::widgets::{Block, Borders, List, ListItem, ListState, Paragraph, Wrap};
use ratatui_core::buffer::Buffer;

use research_utility::sqlite_store::SqliteStore;
use research_utility::sqlite_table_array_store::SqliteTableArrayStore;
use std::collections::hash_map::DefaultHasher;
use tokio::sync::mpsc::{UnboundedReceiver, UnboundedSender, unbounded_channel};

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

const QUESTIONS_PER_PAGE: usize = 10;
const CONVERSATION_SCROLL_SENSITIVITY: usize = 4;

struct QuestionEntry<M: LlmModelMarker, S: DatasetSplit> {
    key: QuestionFlatId<S>,
    action_log: DirectTreeActionLog<M, S>,
    win_rate: f64,
    num_correct: usize,
    num_leaves: usize,
}

enum EntryLoadState<M: LlmModelMarker, S: DatasetSplit> {
    Unloaded,
    Loading,
    Loaded(QuestionEntry<M, S>),
    Failed(String),
}

struct EntryLoadResult<M: LlmModelMarker, S: DatasetSplit> {
    index: usize,
    state: EntryLoadState<M, S>,
}

impl<M: LlmModelMarker, S: DatasetSplit> Clone for QuestionEntry<M, S> {
    fn clone(&self) -> Self {
        Self {
            key: self.key,
            action_log: self.action_log.clone(),
            win_rate: self.win_rate,
            num_correct: self.num_correct,
            num_leaves: self.num_leaves,
        }
    }
}

#[derive(Clone, Copy)]
struct SegmentPosteriorStats {
    pub(super) posterior_mean: f32,
    pub(super) posterior_std: f32,
    pub(super) signal_to_noise: f32,
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

struct TreePage<M: LlmModelMarker, S: DatasetSplit> {
    pub(super) entry_index: usize,
    pub(super) total_actions: usize,
    pub(super) action_limit: usize,
    pub(super) width_division_ratio: usize,
    pub(super) tree_snapshot: TreeDisplaySnapshot<M, S>,
    pub(super) root_segment_id: SegmentId,
    pub(super) segment_advantages_from_posteriors: BTreeMap<SegmentId, f32>,
    pub(super) segment_advantages_from_win_rate: BTreeMap<SegmentId, f32>,
    pub(super) segment_posterior_stats: BTreeMap<SegmentId, SegmentPosteriorStats>,
    pub(super) segment_posterior_signal_scaled: BTreeMap<SegmentId, f32>,
    pub(super) segment_posterior_mean_scaled: BTreeMap<SegmentId, f32>,
    pub(super) segment_posterior_std_scaled: BTreeMap<SegmentId, f32>,
    pub(super) segment_branching_score_display: BTreeMap<SegmentId, Vec<Option<f32>>>,
    pub(super) segment_display_widths: BTreeMap<SegmentId, usize>,
    pub(super) selected_segment_id: SegmentId,
    pub(super) tree_lines: Vec<String>,
    pub(super) rendered_segments: Vec<TreeRenderedSegment>,
    pub(super) hovered_segment_id: Option<SegmentId>,
    pub(super) tree_area: Option<Rect>,
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
    AdvantageFromPosterior,
    AdvantageFromWinRate,
}

impl TreeColorMode {
    fn label(self) -> &'static str {
        match self {
            Self::SignalToNoise => "SignalToNoise",
            Self::BranchingScore => "BranchingScore",
            Self::PosteriorMean => "PosteriorMean",
            Self::PosteriorStd => "PosteriorStd",
            Self::AdvantageFromPosterior => "AdvantageFromPosterior",
            Self::AdvantageFromWinRate => "AdvantageFromWinRate",
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
    pub(super) segment_id: SegmentId,
    pub(super) row: usize,
    pub(super) col: usize,
    pub(super) width: usize,
}

struct ConversationRender {
    pub(super) plain: String,
    pub(super) styled: Text<'static>,
}

struct TreePageState<M: LlmModelMarker, S: DatasetSplit> {
    pub(super) tree_snapshot: TreeDisplaySnapshot<M, S>,
    pub(super) root_segment_id: SegmentId,
    pub(super) segment_advantages_from_posteriors: BTreeMap<SegmentId, f32>,
    pub(super) segment_advantages_from_win_rate: BTreeMap<SegmentId, f32>,
    pub(super) segment_posterior_stats: BTreeMap<SegmentId, SegmentPosteriorStats>,
    pub(super) segment_posterior_signal_scaled: BTreeMap<SegmentId, f32>,
    pub(super) segment_posterior_mean_scaled: BTreeMap<SegmentId, f32>,
    pub(super) segment_posterior_std_scaled: BTreeMap<SegmentId, f32>,
    pub(super) segment_branching_score_display: BTreeMap<SegmentId, Vec<Option<f32>>>,
    pub(super) segment_display_widths: BTreeMap<SegmentId, usize>,
    pub(super) tree_lines: Vec<String>,
    pub(super) rendered_segments: Vec<TreeRenderedSegment>,
}

struct TreeDisplaySnapshot<M: LlmModelMarker, S: DatasetSplit> {
    pub(super) segments: BTreeMap<SegmentId, Segment<M>>,
    pub(super) leaf_segment_judgments: BTreeMap<SegmentId, CorrectnessJudgment>,
    pub(super) _phantom: std::marker::PhantomData<S>,
}

impl<M: LlmModelMarker, S: DatasetSplit> TreeDisplaySnapshot<M, S> {
    fn from_tree(tree: DirectTree<'_, M, S>) -> Self {
        tree.root_segment_id
            .expect("Direct tree browser requires root segment");
        Self {
            pub(super) segments: tree.segments,
            pub(super) leaf_segment_judgments: tree.leaf_segment_judgments,
            pub(super) _phantom: std::marker::PhantomData,
        }
    }

    fn contains_segment(&self, segment_id: SegmentId) -> bool {
        self.segments.contains_key(&segment_id)
    }

    fn get_trajectory_length_till_id(&self, segment_id: SegmentId) -> usize {
        self.path_root_to_segment(segment_id)
            .iter()
            .map(|id| {
                self.segments
                    .get(id)
                    .expect("segment in trajectory must exist")
                    .token_length()
            })
            .sum()
    }

    fn path_root_to_segment(&self, segment_id: SegmentId) -> Vec<SegmentId> {
        let mut path = Vec::new();
        let mut cursor = Some(segment_id);
        while let Some(current_id) = cursor {
            path.push(current_id);
            cursor = self
                .segments
                .get(&current_id)
                .expect("segment in path must exist")
                .parent_id;
        }
        path.reverse();
        path
    }
}

impl<M: LlmModelMarker, S: DatasetSplit> TreePage<M, S> {
    fn new(
        pub(super) entry_index: usize,
        entry: &QuestionEntry<M, S>,
        override_hyperparameters: Option<&PosteriorHyperparameters>,
    ) -> Self {
        let total_actions = entry.action_log.actions.len();
        let action_limit = total_actions;
        let width_division_ratio = width_division_ratio_for_action_log(&entry.action_log);
        let state = tree_page_state_from_action_log(
            &entry.action_log,
            action_limit,
            width_division_ratio,
            override_hyperparameters,
        );
        let selected_segment_id = state.root_segment_id;
        Self {
            entry_index,
            total_actions,
            action_limit,
            width_division_ratio,
            pub(super) tree_snapshot: state.tree_snapshot,
            pub(super) root_segment_id: state.root_segment_id,
            pub(super) segment_advantages_from_posteriors: state.segment_advantages_from_posteriors,
            pub(super) segment_advantages_from_win_rate: state.segment_advantages_from_win_rate,
            pub(super) segment_posterior_stats: state.segment_posterior_stats,
            pub(super) segment_posterior_signal_scaled: state.segment_posterior_signal_scaled,
            pub(super) segment_posterior_mean_scaled: state.segment_posterior_mean_scaled,
            pub(super) segment_posterior_std_scaled: state.segment_posterior_std_scaled,
            pub(super) segment_branching_score_display: state.segment_branching_score_display,
            pub(super) segment_display_widths: state.segment_display_widths,
            selected_segment_id,
            pub(super) tree_lines: state.tree_lines,
            pub(super) rendered_segments: state.rendered_segments,
            pub(super) hovered_segment_id: None,
            pub(super) tree_area: None,
        }
    }

    fn rebuild_snapshot(
        &mut self,
        entry: &QuestionEntry<M, S>,
        override_hyperparameters: Option<&PosteriorHyperparameters>,
    ) {
        let state = tree_page_state_from_action_log::<M, S>(
            &entry.action_log,
            self.action_limit,
            self.width_division_ratio,
            override_hyperparameters,
        );
        self.tree_snapshot = state.tree_snapshot;
        self.root_segment_id = state.root_segment_id;
        self.segment_advantages_from_posteriors = state.segment_advantages_from_posteriors;
        self.segment_advantages_from_win_rate = state.segment_advantages_from_win_rate;
        self.segment_posterior_stats = state.segment_posterior_stats;
        self.segment_posterior_signal_scaled = state.segment_posterior_signal_scaled;
        self.segment_posterior_mean_scaled = state.segment_posterior_mean_scaled;
        self.segment_posterior_std_scaled = state.segment_posterior_std_scaled;
        self.segment_branching_score_display = state.segment_branching_score_display;
        self.segment_display_widths = state.segment_display_widths;
        self.tree_lines = state.tree_lines;
        self.rendered_segments = state.rendered_segments;
        if !self
            .tree_snapshot
            .contains_segment(self.selected_segment_id)
        {
            self.selected_segment_id = self.root_segment_id;
        }
        if self
            .hovered_segment_id
            .is_some_and(|id| !self.tree_snapshot.contains_segment(id))
        {
            self.hovered_segment_id = None;
        }
    }

    fn set_action_limit(
        &mut self,
        entry: &QuestionEntry<M, S>,
        new_limit: usize,
        override_hyperparameters: Option<&PosteriorHyperparameters>,
    ) {
        if self.action_limit == new_limit {
            return;
        }
        self.action_limit = new_limit;
        self.rebuild_snapshot(entry, override_hyperparameters);
    }

    fn set_width_division_ratio(
        &mut self,
        entry: &QuestionEntry<M, S>,
        new_ratio: usize,
        override_hyperparameters: Option<&PosteriorHyperparameters>,
    ) {
        if self.width_division_ratio == new_ratio {
            return;
        }
        self.width_division_ratio = new_ratio.max(1);
        self.rebuild_snapshot(entry, override_hyperparameters);
    }
}

fn tree_page_state_from_action_log<M: LlmModelMarker, S: DatasetSplit>(
    action_log: &DirectTreeActionLog<M, S>,
    pub(super) action_limit: usize,
    pub(super) width_division_ratio: usize,
    override_hyperparameters: Option<&PosteriorHyperparameters>,
) -> TreePageState<M, S> {
    let partial_log = partial_action_log(action_log, action_limit);
    let tree = DirectTree::<M, S>::from_action_log(&partial_log);
    let root_segment_id = tree_root_segment_id(&tree);
    let segment_display_widths = segment_display_widths(&tree, Some(width_division_ratio));
    let segment_advantages_from_posteriors =
        segment_advantages_from_posteriors(&tree, override_hyperparameters);
    let segment_advantages_from_win_rate = segment_advantages_from_win_rate(&tree);
    let segment_posterior_stats = segment_posterior_stats(&tree, override_hyperparameters);
    let segment_posterior_signal_scaled =
        scaled_segment_posterior_signal(&tree, &segment_posterior_stats);
    let segment_posterior_mean_scaled =
        scaled_segment_posterior_mean(&tree, &segment_posterior_stats);
    let segment_posterior_std_scaled =
        scaled_segment_posterior_std(&tree, &segment_posterior_stats);
    let segment_branching_score_display = segment_branching_score_display(
        &tree,
        width_division_ratio,
        &segment_display_widths,
        override_hyperparameters,
    );
    let (tree_lines, rendered_segments) = build_segment_graph_lines(
        &tree,
        root_segment_id,
        &tree.leaf_segment_judgments,
        &segment_display_widths,
    );
    let tree_snapshot = TreeDisplaySnapshot::from_tree(tree);
    TreePageState {
        tree_snapshot,
        root_segment_id,
        segment_advantages_from_posteriors,
        segment_advantages_from_win_rate,
        segment_posterior_stats,
        segment_posterior_signal_scaled,
        segment_posterior_mean_scaled,
        segment_posterior_std_scaled,
        segment_branching_score_display,
        segment_display_widths,
        tree_lines,
        rendered_segments,
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

fn fixed_width_scale_ratio_for_tree<M: crate::llm_model::LlmModelMarker, S: DatasetSplit>(
    tree: &DirectTree<'_, M, S>,
) -> f64 {
    let trunk_trajectory_lengths: Vec<usize> = tree
        .trunk_leaf_segments
        .iter()
        .map(|leaf_id| *&tree.get_trajectory_length_till_id(*leaf_id))
        .collect();

    let avg_trunk_trajectory_len = if trunk_trajectory_lengths.is_empty() {
        1.0
    } else {
        trunk_trajectory_lengths
            .iter()
            .map(|value| *value as f64)
            .sum::<f64>()
            / trunk_trajectory_lengths.len() as f64
    };
    assert!(avg_trunk_trajectory_len > 0.0);
    100.0_f64 / avg_trunk_trajectory_len
}

fn segment_display_widths<M: crate::llm_model::LlmModelMarker, S: DatasetSplit>(
    tree: &DirectTree<'_, M, S>,
    pub(super) width_division_ratio: Option<usize>,
) -> BTreeMap<SegmentId, usize> {
    let mut widths = BTreeMap::new();
    if tree.segments.is_empty() {
        return widths;
    }

    let division_ratio =
        width_division_ratio.unwrap_or_else(|| width_division_ratio_for_tree(tree));

    for (segment_id, segment) in &tree.segments {
        let token_len = segment.token_length();
        let scaled = token_len / division_ratio.max(1);
        widths.insert(*segment_id, scaled.max(1));
    }
    widths
}

fn segment_branching_score_display<M: crate::llm_model::LlmModelMarker, S: DatasetSplit>(
    tree: &DirectTree<'_, M, S>,
    pub(super) width_division_ratio: usize,
    pub(super) segment_display_widths: &BTreeMap<SegmentId, usize>,
    override_hyperparameters: Option<&PosteriorHyperparameters>,
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
    let per_token_branching_scores: BTreeMap<
        SegmentId,
        BTreeMap<ContentIndex, BTreeMap<usize, TokenBranchingScore>>,
    > = tree.calculate_per_token_branching_scores(&segment_uncertainty_scores);
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

fn segment_posterior_stats<M: crate::llm_model::LlmModelMarker, S: DatasetSplit>(
    tree: &DirectTree<'_, M, S>,
    override_hyperparameters: Option<&PosteriorHyperparameters>,
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
                pub(super) posterior_mean: posterior.mean,
                posterior_std,
                signal_to_noise,
            },
        );
    }
    stats_by_segment
}

fn scaled_segment_posterior_signal<M: crate::llm_model::LlmModelMarker, S: DatasetSplit>(
    tree: &DirectTree<'_, M, S>,
    stats_by_segment: &BTreeMap<SegmentId, SegmentPosteriorStats>,
) -> BTreeMap<SegmentId, f32> {
    scaled_segment_posterior_value(tree, stats_by_segment, |stats| stats.signal_to_noise)
}

fn scaled_segment_posterior_mean<M: crate::llm_model::LlmModelMarker, S: DatasetSplit>(
    tree: &DirectTree<'_, M, S>,
    stats_by_segment: &BTreeMap<SegmentId, SegmentPosteriorStats>,
) -> BTreeMap<SegmentId, f32> {
    scaled_segment_posterior_value(tree, stats_by_segment, |stats| stats.posterior_mean)
}

fn scaled_segment_posterior_std<M: crate::llm_model::LlmModelMarker, S: DatasetSplit>(
    tree: &DirectTree<'_, M, S>,
    stats_by_segment: &BTreeMap<SegmentId, SegmentPosteriorStats>,
) -> BTreeMap<SegmentId, f32> {
    scaled_segment_posterior_value(tree, stats_by_segment, |stats| stats.posterior_std)
}

fn scaled_segment_posterior_value<M: crate::llm_model::LlmModelMarker, S: DatasetSplit>(
    tree: &DirectTree<'_, M, S>,
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

pub(super) fn posterior_stat_min_max(
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

fn advantage_to_color(advantage: f32) -> Color {
    let clamped = advantage.clamp(-3.0, 3.0);
    signed_posterior_to_color(clamped / 3.0)
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
    pub(super) text_hash: u64,
    pub(super) width: u16,
    pub(super) height: u16,
    pub(super) lines: usize,
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
        pub(super) width: area.width,
        pub(super) height: area.height,
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

fn collect_leaf_order<M: LlmModelMarker, S: DatasetSplit>(
    tree: &DirectTree<'_, M, S>,
    pub(super) root_segment_id: SegmentId,
) -> Vec<SegmentId> {
    let mut leaves = Vec::new();
    let mut stack = vec![root_segment_id];
    while let Some(segment_id) = stack.pop() {
        let segment = tree
            .segments
            .get(&segment_id)
            .expect("segment in tree must exist");
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

fn path_root_to_segment<M: LlmModelMarker, S: DatasetSplit>(
    tree: &DirectTree<'_, M, S>,
    pub(super) segment_id: SegmentId,
) -> Vec<SegmentId> {
    let mut path = Vec::new();
    let mut cursor = Some(segment_id);
    while let Some(current_id) = cursor {
        path.push(current_id);
        cursor = tree
            .segments
            .get(&current_id)
            .expect("segment in path must exist")
            .parent_id;
    }
    path.reverse();
    path
}

fn build_segment_graph_lines<M: LlmModelMarker, S: DatasetSplit>(
    tree: &DirectTree<'_, M, S>,
    pub(super) root_segment_id: SegmentId,
    pub(super) leaf_segment_judgments: &BTreeMap<SegmentId, CorrectnessJudgment>,
    pub(super) segment_display_widths: &BTreeMap<SegmentId, usize>,
) -> (Vec<String>, Vec<TreeRenderedSegment>) {
    let ordered_leaf_ids = collect_leaf_order(tree, root_segment_id);
    let line_count = ordered_leaf_ids.len().max(1);
    let mut canvas: Vec<Vec<char>> = vec![Vec::new(); line_count];

    let mut row_by_segment: BTreeMap<SegmentId, usize> = BTreeMap::new();
    for (row, leaf_id) in ordered_leaf_ids.iter().copied().enumerate() {
        for segment_id in path_root_to_segment(tree, leaf_id) {
            row_by_segment.entry(segment_id).or_insert(row);
        }
    }

    if row_by_segment.is_empty() {
        row_by_segment.insert(root_segment_id, 0);
    }

    let mut width_by_segment: BTreeMap<SegmentId, usize> = BTreeMap::new();
    for segment_id in tree.segments.keys().copied() {
        let width = segment_display_widths
            .get(&segment_id)
            .copied()
            .unwrap_or(1)
            .max(1);
        width_by_segment.insert(segment_id, width);
    }

    let mut col_by_segment: BTreeMap<SegmentId, usize> = BTreeMap::new();
    col_by_segment.insert(root_segment_id, 0);
    let mut stack = vec![root_segment_id];
    while let Some(parent_id) = stack.pop() {
        let parent_col = *col_by_segment
            .get(&parent_id)
            .expect("parent column must exist");
        let parent_width = *width_by_segment
            .get(&parent_id)
            .expect("parent width must exist");
        let child_col = parent_col + parent_width + 3;
        let parent = tree
            .segments
            .get(&parent_id)
            .expect("parent segment must exist in tree");
        for child_id in parent.child_ids.iter().rev() {
            if col_by_segment.contains_key(child_id) {
                continue;
            }
            col_by_segment.insert(*child_id, child_col);
            stack.push(*child_id);
        }
    }
    for segment_id in tree.segments.keys().copied() {
        if col_by_segment.contains_key(&segment_id) {
            continue;
        }
        let path = path_root_to_segment(tree, segment_id);
        let mut col = 0usize;
        for parent_id in path.iter().take(path.len().saturating_sub(1)) {
            col = col
                .saturating_add(width_by_segment.get(parent_id).copied().unwrap_or(1))
                .saturating_add(3);
        }
        col_by_segment.insert(segment_id, col);
    }

    let mut rendered_segments = Vec::new();
    for segment_id in tree.segments.keys().copied() {
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

    for (parent_id, parent) in &tree.segments {
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
        let suffix = if let Some(judgment) = leaf_segment_judgments.get(&leaf_id) {
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

fn render_tree_line<M: LlmModelMarker, S: DatasetSplit>(
    tree_page: &TreePage<M, S>,
    pub(super) row: usize,
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
            TreeColorMode::AdvantageFromPosterior => {
                let advantage = tree_page
                    .segment_advantages_from_posteriors
                    .get(&rendered.segment_id)
                    .copied()
                    .unwrap_or(0.0);
                let mut style = Style::default().fg(advantage_to_color(advantage));
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
            TreeColorMode::AdvantageFromWinRate => {
                let advantage = tree_page
                    .segment_advantages_from_win_rate
                    .get(&rendered.segment_id)
                    .copied()
                    .unwrap_or(0.0);
                let mut style = Style::default().fg(advantage_to_color(advantage));
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

fn tree_segment_at_mouse<M: LlmModelMarker, S: DatasetSplit>(
    tree_page: &TreePage<M, S>,
    column: u16,
    pub(super) row: u16,
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
    pub(super) lines: &mut Vec<Vec<Span<'static>>>,
    text: &str,
    style: Option<Style>,
    pub(super) plain: &mut String,
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

fn build_conversation_render<M: LlmModelMarker, S: DatasetSplit>(
    pub(super) tree_snapshot: &TreeDisplaySnapshot<M, S>,
    pub(super) segment_posterior_signal_scaled: &BTreeMap<SegmentId, f32>,
    pub(super) segment_id: SegmentId,
) -> ConversationRender {
    let path = tree_snapshot.path_root_to_segment(segment_id);

    let mut plain = String::new();
    let mut lines: Vec<Vec<Span<'static>>> = vec![Vec::new()];
    for sid in path {
        let segment = tree_snapshot
            .segments
            .get(&sid)
            .expect("segment in path must exist");
        let segment_color = segment_posterior_signal_scaled
            .get(&sid)
            .copied()
            .map(signed_posterior_to_color)
            .unwrap_or(Color::White);
        let reasoning_style = Some(Style::default().fg(segment_color));
        for content in &segment.content {
            match content {
                SegmentContent::Prompt(tokens) => {
                    push_text_as_spans(&mut lines, &tokens.decode(), None, &mut plain);
                }
                SegmentContent::ReasoningOrToolCall {
                    tokens,
                    complete: _,
                } => {
                    push_text_as_spans(&mut lines, &tokens.decode(), reasoning_style, &mut plain);
                }
                SegmentContent::ToolResponse(tokens) => {
                    push_text_as_spans(&mut lines, &tokens.decode(), None, &mut plain);
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
        pub(super) styled: Text::from(styled_lines),
    }
}

fn partial_action_log<M: LlmModelMarker, S: DatasetSplit>(
    action_log: &DirectTreeActionLog<M, S>,
    pub(super) action_limit: usize,
) -> DirectTreeActionLog<M, S> {
    let mut partial_log = action_log.clone();
    partial_log.actions = action_log
        .actions
        .iter()
        .take(action_limit)
        .cloned()
        .collect();
    partial_log
}

fn tree_root_segment_id<M: LlmModelMarker, S: DatasetSplit>(
    tree: &DirectTree<'_, M, S>,
) -> SegmentId {
    tree.root_segment_id
        .expect("Direct tree browser requires root segment")
}

fn segment_advantages_from_posteriors<M: LlmModelMarker, S: DatasetSplit>(
    tree: &DirectTree<'_, M, S>,
    override_hyperparameters: Option<&PosteriorHyperparameters>,
) -> BTreeMap<SegmentId, f32> {
    let mut advantages =
        tree.calculate_segment_advantages_from_posteriors(override_hyperparameters);
    for segment_id in tree.segments.keys().copied() {
        advantages.entry(segment_id).or_insert(0.0);
    }
    advantages
}

fn segment_advantages_from_win_rate<M: LlmModelMarker, S: DatasetSplit>(
    tree: &DirectTree<'_, M, S>,
) -> BTreeMap<SegmentId, f32> {
    let mut advantages = tree.calculate_segment_advantages_from_win_rate();
    for segment_id in tree.segments.keys().copied() {
        advantages.entry(segment_id).or_insert(0.0);
    }
    advantages
}

fn width_division_ratio_for_action_log<M: LlmModelMarker, S: DatasetSplit>(
    action_log: &DirectTreeActionLog<M, S>,
) -> usize {
    width_division_ratio_for_tree(&DirectTree::<M, S>::from_action_log(action_log))
}

fn width_division_ratio_for_tree<M: LlmModelMarker, S: DatasetSplit>(
    tree: &DirectTree<'_, M, S>,
) -> usize {
    let fixed_scale_ratio = fixed_width_scale_ratio_for_tree(tree);
    if fixed_scale_ratio <= 0.0 {
        return 1;
    }
    ((1.0 / fixed_scale_ratio).round() as usize).max(1)
}

fn question_stats_from_action_log<M: LlmModelMarker, S: DatasetSplit>(
    action_log: &DirectTreeActionLog<M, S>,
) -> (usize, usize, f64) {
    let final_tree = DirectTree::<M, S>::from_action_log(action_log);
    let num_leaves = final_tree.leaf_segment_judgments.len();
    let num_correct = final_tree
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
