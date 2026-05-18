use std::collections::BTreeMap;

use crate::{
    direct_tool::direct_tree::{DirectTree, SegmentId},
    llm_model::LlmModelMarker,
};

#[derive(Debug, Clone)]
pub struct Posterior {
    pub mean: f32,
    pub log_std: f32,
}

impl<M: LlmModelMarker> DirectTree<M> {
    pub fn calculate_segment_posteriors(&self) -> BTreeMap<SegmentId, Posterior> {
        todo!()
    }
}
