use serde::{Deserialize, Serialize};

use crate::agent::{single_dataset::SingleDatasetQuestion, tree_action::TreeAction};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct TreeActionLog {
    pub question: SingleDatasetQuestion,
    pub actions: Vec<TreeAction>,
}
