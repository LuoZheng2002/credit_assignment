use serde::{Deserialize, Serialize};

// this tree is similar to the completed tree in src/agent folder, but now it runs on a lightweight tool-calling context instead of a heavy agent framework
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct DirectTree {
    pub flat_id: usize, // the same flat id as the one in the hybrid dataset
    pub dataset_name: String,
    pub question_id: usize,
    pub question: String,
    pub correct_answer: String,
    pub segments: Vec<Segment>,
}

// it has interleaved reasoning and tool response
// we can branch on the reasoning part, but not on the tool response part
// tool response should not be counted towards the segment length
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Segment {
    pub segment_id: usize,
    pub content: Vec<SegmentContent>,
    pub child_ids: Vec<usize>,
    pub parent_id: Option<usize>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum SegmentContent {
    ReasoningOrToolCall(String),
    ToolResponse(String),
}
