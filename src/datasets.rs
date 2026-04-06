use serde::{Deserialize, Serialize};

use crate::parallel_process_jsonl::HasId;

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct DeepMathQuestion {
    pub id: usize,
    pub question: String,
    pub final_answer: String,
}

impl HasId for DeepMathQuestion {
    fn id(&self) -> usize {
        self.id
    }
}


#[derive(Serialize, Deserialize, Debug)]
pub struct DeepMathQuestionReasoning {
    pub id: usize,
    pub reasoning: String,
    pub final_answer: String,
    pub question: String,
}

impl HasId for DeepMathQuestionReasoning {
    fn id(&self) -> usize {
        self.id
    }
}

pub fn get_deepmath_questions_path(num_samples: usize) -> String {
    format!("datasets/deepmath_samples_{}.jsonl", num_samples)
}

pub fn get_deepmath_questions_with_reasoning_path(num_samples: usize) -> String {
    format!("datasets/deepmath_samples_{}_reasoning.jsonl", num_samples)
}

