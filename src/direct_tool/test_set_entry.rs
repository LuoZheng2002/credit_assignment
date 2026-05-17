use serde::{Deserialize, Serialize};

use crate::json_line_util::HasId;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestSetEntry {
    pub flat_id: usize,
    pub question_id: usize,
    pub question: String,
    pub correct_answer: String,
}

impl HasId for TestSetEntry {
    fn id(&self) -> usize {
        self.flat_id
    }
}
