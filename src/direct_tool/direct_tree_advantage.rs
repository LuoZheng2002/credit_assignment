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
        Self::segment_advantages_from_posteriors(posteriors)
    }
    fn segment_advantages_from_posteriors(
        posteriors: BTreeMap<SegmentId, Posterior>,
    ) -> BTreeMap<SegmentId, f32> {
        let segment_advantages_unnormalized = posteriors
            .into_iter()
            .map(|(segment_id, posterior)| {
                (segment_id, posterior.mean / posterior.log_std.exp()) // we use the mean/std as the unnormalized advantage
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
