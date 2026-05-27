use std::collections::BTreeMap;

use crate::{
    direct_tool::{
        direct_tree::{DirectTree, SegmentId},
        direct_tree_posterior::Posterior,
        posterior_calculation_config::PosteriorHyperparameters,
    },
    llm_model::LlmModelMarker,
};

impl<M: LlmModelMarker> DirectTree<M> {
    pub fn calculate_segment_advantages(
        &self,
        override_hyperparameters: Option<PosteriorHyperparameters>,
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
}
