use crate::{
    agent::{action_log_schema::TreeActionLogStore, tree_action_log::TreeActionLog},
    llm_model::LlmModelName,
};

pub type SqliteSessionLogStore = TreeActionLogStore;
pub type SqliteSessionLogEntry = TreeActionLog;

pub fn get_rollout_log_path(model: LlmModelName, dataset_name: &str, num_samples: usize) -> String {
    format!(
        "results/{}/agent/{}_rollout_log_{}.sqlite",
        model.cli_name(),
        dataset_name,
        num_samples
    )
}
