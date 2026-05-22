use crate::{direct_tool::{
    direct_tree::DirectTree, direct_tree_action_log::DirectTreeActionLog, posterior_schema::{PosteriorCalculationConfig, PosteriorFitPerTree}
}, llm_model::LlmModelMarker};

pub fn action_log_to_posterior_fit<M: LlmModelMarker>(
    action_log: &DirectTreeActionLog,
    posterior_calculation_config: &PosteriorCalculationConfig,
) -> PosteriorFitPerTree {
    let tree = DirectTree::<M>::from_action_log(action_log);
    todo!()
}
