use crate::{agent::tree_action::TreeAction, direct_answer::generate_raw_answers::LlmModel};

pub type SqliteSessionLogStore =
    research_utility::sqlite_table_array_store::SqliteTableArrayStore<usize, TreeAction>;

pub fn get_rollout_log_path(model: LlmModel, dataset_name: &str, num_samples: usize) -> String {
    format!(
        "results/{}/agent/{}_rollout_log_{}.sqlite",
        model.cli_name(),
        dataset_name,
        num_samples
    )
}
