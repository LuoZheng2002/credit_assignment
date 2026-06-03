use std::collections::{BTreeMap, BTreeSet};

use crate::{
    direct_tool::{
        direct_tree::{DirectTree, SegmentId},
        direct_tree_posterior::Posterior,
        hybrid_dataset::DatasetSplit,
        posterior_calculation_config::PosteriorHyperparameters,
    },
    llm_model::LlmModelMarker,
};

#[derive(Debug, Clone)]
pub struct WinRate {
    pub num_wins: usize,
    pub total_plays: usize,
}

impl<'a, M: LlmModelMarker, S: DatasetSplit> DirectTree<'a, M, S> {
    pub fn calculate_segment_advantages_from_posteriors(
        &self,
        override_hyperparameters: Option<&PosteriorHyperparameters>,
    ) -> BTreeMap<SegmentId, f32> {
        let posteriors = self.calculate_segment_posteriors(override_hyperparameters);
        self.segment_advantages_from_posteriors(posteriors)
    }
    fn segment_advantages_from_posteriors(
        &self,
        posteriors: BTreeMap<SegmentId, Posterior>,
    ) -> BTreeMap<SegmentId, f32> {
        let longest_reasoning_only_token_length = self
            .segments
            .values()
            .map(|segment| segment.reasoning_only_token_length())
            .max()
            .unwrap_or(0);
        let min_clamped_reasoning_only_token_length =
            (longest_reasoning_only_token_length as f32 / 8.0).max(1.0);

        let segment_advantages_unnormalized = posteriors
            .into_iter()
            .map(|(segment_id, posterior)| {
                let segment = self
                    .segments
                    .get(&segment_id)
                    .expect("Posterior segment id must exist in tree segments");
                let clamped_reasoning_only_token_length = (segment.reasoning_only_token_length()
                    as f32)
                    .max(min_clamped_reasoning_only_token_length);
                (
                    segment_id,
                    (posterior.mean / posterior.log_std.exp())
                        / clamped_reasoning_only_token_length,
                )
            })
            .collect::<Vec<(SegmentId, f32)>>();
        if segment_advantages_unnormalized.is_empty() {
            return BTreeMap::new();
        }
        let segment_advantages_mean = segment_advantages_unnormalized
            .iter()
            .map(|(_, advantage)| *advantage)
            .sum::<f32>()
            / segment_advantages_unnormalized.len() as f32;
        let segment_advantages_std = (segment_advantages_unnormalized
            .iter()
            .map(|(_, advantage)| {
                let diff = *advantage - segment_advantages_mean;
                diff * diff
            })
            .sum::<f32>()
            / segment_advantages_unnormalized.len() as f32)
            .sqrt();
        let segment_advantages = segment_advantages_unnormalized
            .into_iter()
            .map(|(segment_id, advantage)| {
                let normalized_advantage = if segment_advantages_std > 0.0 {
                    let mut shifted_advantage = advantage - segment_advantages_mean;
                    if shifted_advantage * advantage < 0.0 {
                        shifted_advantage = 0.0; // if the advantage is on the opposite side of the mean compared to the unnormalized advantage, we set it to 0 to avoid hurting the training
                    }
                    shifted_advantage / segment_advantages_std
                } else {
                    0.0
                };
                (segment_id, normalized_advantage)
            })
            .collect::<BTreeMap<SegmentId, f32>>();
        segment_advantages
    }
    pub fn calculate_segment_advantages_from_win_rate(&self) -> BTreeMap<SegmentId, f32> {
        let mut segment_win_rate_memo: BTreeMap<SegmentId, WinRate> = BTreeMap::new();
        let mut segment_ids: BTreeSet<SegmentId> = self.segments.keys().cloned().collect();
        segment_ids.remove(&self.root_segment_id.expect("The tree must have a root id"));
        for segment_id in segment_ids {
            self.find_segment_win_rate(segment_id, &mut segment_win_rate_memo);
        }
        let advantage_unnormalized = segment_win_rate_memo
            .into_iter()
            .map(|(segment_id, win_rate)| {
                assert!(win_rate.total_plays > 0);
                let win_rate_value = win_rate.num_wins as f32 / win_rate.total_plays as f32;
                (segment_id, win_rate_value)
            })
            .collect::<Vec<(SegmentId, f32)>>();
        let advantage_mean = advantage_unnormalized
            .iter()
            .map(|(_, advantage)| *advantage)
            .sum::<f32>()
            / advantage_unnormalized.len() as f32;
        let advantage_std = (advantage_unnormalized
            .iter()
            .map(|(_, advantage)| {
                let diff = *advantage - advantage_mean;
                diff * diff
            })
            .sum::<f32>()
            / advantage_unnormalized.len() as f32)
            .sqrt();
        let advantage_normalized: BTreeMap<SegmentId, f32> = advantage_unnormalized
            .iter()
            .map(|(segment_id, advantage)| {
                let normalized_advantage = if advantage_std > 0.0 {
                    (*advantage - advantage_mean) / advantage_std
                } else {
                    0.0
                };
                (*segment_id, normalized_advantage)
            })
            .collect();
        advantage_normalized
    }

    fn find_segment_win_rate(
        &self,
        segment_id: SegmentId,
        memo: &mut BTreeMap<SegmentId, WinRate>,
    ) -> WinRate {
        if let Some(win_rate) = memo.get(&segment_id) {
            return win_rate.clone();
        }
        let segment = self
            .segments
            .get(&segment_id)
            .expect("Segment id must exist in tree segments");
        if segment.child_ids.is_empty() {
            let judgment = self
                .leaf_segment_judgments
                .get(&segment_id)
                .expect("Leaf segment must have judgment");
            let num_wins = if judgment.is_correct { 1 } else { 0 };
            let total_plays = 1;
            let win_rate = WinRate {
                num_wins,
                total_plays,
            };
            memo.insert(segment_id, win_rate.clone());
            return win_rate;
        } else {
            // non-leaf node, we need to aggregate the win/loss from its children
            let mut num_wins = 0;
            let mut total_plays = 0;
            for child_id in &segment.child_ids {
                let child_win_rate = self.find_segment_win_rate(*child_id, memo);
                num_wins += child_win_rate.num_wins;
                total_plays += child_win_rate.total_plays;
            }
            let win_rate = WinRate {
                num_wins,
                total_plays,
            };
            memo.insert(segment_id, win_rate.clone());
            return win_rate;
        }
    }
}
