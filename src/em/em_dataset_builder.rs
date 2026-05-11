use std::collections::BTreeSet;

use crate::agent::tree::Tree;

use super::em_types::{
    EmFitDataset, EmLeafBinding, EmLeafPath, EmNodeBinding, LeafLabel, SparsePathTerm,
};

/// Builds EM fitting datasets from rollout trees.
#[derive(Debug, Default, Clone)]
pub struct EmDatasetBuilder;

impl EmDatasetBuilder {
    pub fn new() -> Self {
        Self
    }

    pub fn build_from_trees(&self, trees: &[Tree]) -> EmFitDataset {
        assert!(!trees.is_empty(), "EM dataset build requires at least one tree");

        let mut node_bindings: Vec<EmNodeBinding> = Vec::new();
        let mut leaf_bindings: Vec<EmLeafBinding> = Vec::new();
        let mut leaf_paths: Vec<EmLeafPath> = Vec::new();

        for tree in trees {
            assert!(
                tree.root_node_id.is_some(),
                "Tree {} must have a root node",
                tree.question_id
            );
            let root_node_id = tree
                .root_node_id
                .expect("Tree must have root_node_id before EM dataset build");
            assert_eq!(
                root_node_id, 0,
                "Tree {} root node id must be 0",
                tree.question_id
            );
            assert!(
                !tree.nodes.is_empty(),
                "Tree {} must have at least one node",
                tree.question_id
            );
            assert_eq!(
                tree.leaf_node_ids.len(),
                tree.leaf_node_judgments.len(),
                "Tree {} requires matching counts of leaf ids and leaf judgments",
                tree.question_id
            );

            let leaf_ids_set: BTreeSet<usize> = tree.leaf_node_ids.iter().copied().collect();
            let leaf_judgment_ids_set: BTreeSet<usize> =
                tree.leaf_node_judgments.keys().copied().collect();
            assert_eq!(
                leaf_ids_set, leaf_judgment_ids_set,
                "Tree {} requires leaf ids and leaf judgment keys to be identical sets",
                tree.question_id
            );

            let node_global_id_start = node_bindings.len();
            let mut local_to_global_node_id: Vec<usize> = vec![0; tree.nodes.len()];
            for node in &tree.nodes {
                assert!(
                    node.node_id < tree.nodes.len(),
                    "Tree {} node_id {} out of bounds for nodes length {}",
                    tree.question_id,
                    node.node_id,
                    tree.nodes.len()
                );
                assert_eq!(
                    tree.nodes[node.node_id].node_id,
                    node.node_id,
                    "Tree {} node index must equal node_id",
                    tree.question_id
                );
                if let Some(parent_id) = node.parent_id {
                    assert!(
                        parent_id < tree.nodes.len(),
                        "Tree {} parent_id {} out of bounds for nodes length {}",
                        tree.question_id,
                        parent_id,
                        tree.nodes.len()
                    );
                }

                let global_node_id = node_global_id_start + node.node_id;
                local_to_global_node_id[node.node_id] = global_node_id;
                node_bindings.push(EmNodeBinding {
                    global_node_id,
                    tree_question_id: tree.question_id,
                    node_id: node.node_id,
                    node_type: node.step.node_type(),
                });
            }

            for &leaf_node_id in &tree.leaf_node_ids {
                let judgment = tree.leaf_node_judgments.get(&leaf_node_id).expect(
                    "Leaf id in leaf_node_ids must exist in leaf_node_judgments during EM build",
                );
                let label = if judgment.is_correct {
                    LeafLabel::Correct
                } else {
                    LeafLabel::Incorrect
                };

                let global_leaf_id = leaf_bindings.len();
                leaf_bindings.push(EmLeafBinding {
                    global_leaf_id,
                    tree_question_id: tree.question_id,
                    leaf_node_id,
                    label,
                });

                let mut node_ids_from_leaf_to_root: Vec<usize> = Vec::new();
                let mut seen_on_path: BTreeSet<usize> = BTreeSet::new();
                let mut cursor = Some(leaf_node_id);
                while let Some(node_id) = cursor {
                    assert!(
                        node_id < tree.nodes.len(),
                        "Tree {} path node_id {} out of bounds for nodes length {}",
                        tree.question_id,
                        node_id,
                        tree.nodes.len()
                    );
                    assert!(
                        seen_on_path.insert(node_id),
                        "Tree {} leaf path to node {} contains a duplicated node id",
                        tree.question_id,
                        leaf_node_id
                    );
                    let node = tree.get_node_by_id(node_id);
                    node_ids_from_leaf_to_root.push(node_id);
                    cursor = node.parent_id;
                }

                assert!(
                    !node_ids_from_leaf_to_root.is_empty(),
                    "Tree {} leaf node {} must yield a non-empty root path",
                    tree.question_id,
                    leaf_node_id
                );

                node_ids_from_leaf_to_root.reverse();
                assert_eq!(
                    node_ids_from_leaf_to_root[0], root_node_id,
                    "Tree {} leaf node {} path must start at root",
                    tree.question_id, leaf_node_id
                );

                let terminal_node_id = *node_ids_from_leaf_to_root
                    .last()
                    .expect("Leaf path should have terminal node after non-empty assertion");
                assert_eq!(
                    terminal_node_id, leaf_node_id,
                    "Tree {} leaf path terminal node must equal leaf node id",
                    tree.question_id
                );

                let terms: Vec<SparsePathTerm> = node_ids_from_leaf_to_root
                    .into_iter()
                    .map(|node_id| SparsePathTerm {
                        global_node_id: local_to_global_node_id[node_id],
                        x_li: 1.0,
                    })
                    .collect();
                leaf_paths.push(EmLeafPath {
                    global_leaf_id,
                    terms,
                });
            }
        }

        assert!(
            !leaf_bindings.is_empty(),
            "EM dataset build requires at least one judged leaf across trees"
        );
        assert_eq!(
            leaf_paths.len(),
            leaf_bindings.len(),
            "EM dataset build requires one sparse path per judged leaf"
        );
        for (index, leaf_path) in leaf_paths.iter().enumerate() {
            assert_eq!(
                leaf_path.global_leaf_id, index,
                "leaf_paths must be aligned by global_leaf_id"
            );
        }

        EmFitDataset {
            node_bindings,
            leaf_bindings,
            leaf_paths,
        }
    }
}
