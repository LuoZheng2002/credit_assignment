use serde::{Deserialize, Serialize};

use crate::{
    hybrid_dataset::DatasetSplit,
    tree::{ContentIndex, DirectTree, SegmentContent, SegmentId},
    tree_action::TokenPositionInTree,
    llm_model::LlmModelMarker,
};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct TokenPositionInSegment {
    pub content_index: ContentIndex,
    pub offset: usize,
}
pub struct LongestCommonPrefixResult {
    pub common_prefix_length: usize,
    pub diverge_position_in_tree: TokenPositionInTree,
    pub diverge_position_in_query: TokenPositionInSegment,
}

impl<'a, M: LlmModelMarker, S: DatasetSplit> DirectTree<'a, M, S> {
    // query does not include the root segment
    pub fn find_longest_common_prefix(
        &self,
        query: &[SegmentContent<M>],
    ) -> LongestCommonPrefixResult {
        let root_segment = self
            .segments
            .get(&self.root_segment_id.unwrap())
            .expect("Root segment not found");
        let mut flattened_query: Vec<(i32, TokenPositionInSegment)> = vec![];
        for (content_index, content) in query.iter().enumerate() {
            let tokens = content.tokens();
            for (token_offset, token) in tokens.into_iter().enumerate() {
                flattened_query.push((
                    token,
                    TokenPositionInSegment {
                        content_index,
                        offset: token_offset,
                    },
                ));
            }
        }
        let mut common_prefixes: Vec<LongestCommonPrefixResult> = vec![];
        for child_id in &root_segment.child_ids {
            common_prefixes.push(self.find_longest_common_prefix_helper(
                &flattened_query,
                *child_id,
                0,
            ));
        }
        common_prefixes
            .into_iter()
            .max_by_key(|res| res.common_prefix_length)
            .expect("There should be at least one child segment under the root, otherwise we should not do spontaneous branching")
    }
    fn find_longest_common_prefix_helper(
        &self,
        flattened_query: &[(i32, TokenPositionInSegment)],
        current_tree_segment_id: SegmentId,
        query_token_offset: usize,
    ) -> LongestCommonPrefixResult {
        let segment = self
            .segments
            .get(&current_tree_segment_id)
            .expect("Segment not found in tree");
        let mut segment_contents: Vec<(i32, TokenPositionInSegment)> = vec![];
        for (content_index, content) in segment.content.iter().enumerate() {
            let tokens = content.tokens();
            for (token_offset, token) in tokens.into_iter().enumerate() {
                segment_contents.push((
                    token,
                    TokenPositionInSegment {
                        content_index,
                        offset: token_offset,
                    },
                ));
            }
        }
        let prefix_result_at_index = |index: usize| LongestCommonPrefixResult {
            common_prefix_length: query_token_offset + index,
            diverge_position_in_tree: TokenPositionInTree {
                segment_id: current_tree_segment_id,
                content_index: segment_contents[index].1.content_index,
                offset: segment_contents[index].1.offset,
            },
            diverge_position_in_query: TokenPositionInSegment {
                content_index: flattened_query[query_token_offset + index].1.content_index,
                offset: flattened_query[query_token_offset + index].1.offset,
            },
        };
        for index in 0.. {
            // check if we have reached the end of the query or the segment
            let query_sum_offset = query_token_offset + index;
            // let mut return_current_position = false;
            if query_sum_offset >= flattened_query.len() {
                // we assert that index > 0, otherwise it will be handled at the end of this function
                assert!(index > 0);
                return prefix_result_at_index(index - 1);
            }
            if index >= segment_contents.len() {
                // current tree segment is exhausted
                assert!(query_sum_offset < flattened_query.len()); // it is already handled above
                // then we need to recursively call the helper on the child segments
                let mut common_prefixes: Vec<LongestCommonPrefixResult> = vec![];
                for child_id in &segment.child_ids {
                    common_prefixes.push(self.find_longest_common_prefix_helper(
                        flattened_query,
                        *child_id,
                        query_sum_offset,
                    ));
                }
                return common_prefixes
                    .into_iter()
                    .max_by_key(|res| res.common_prefix_length)
                    .unwrap_or_else(|| {
                        prefix_result_at_index(index - 1) // we make sure that there is at least one token that is different
                    });
            }
            if flattened_query[query_sum_offset].0 != segment_contents[index].0 {
                return prefix_result_at_index(index);
            }
        }
        unreachable!()
    }
}
