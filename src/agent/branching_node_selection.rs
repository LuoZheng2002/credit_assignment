use rand::distr::{Distribution, weighted::WeightedIndex};

use crate::agent::{
    tool_call_execution::MAX_NUM_TRAJECTORIES, trajectory_action_types::NodeType,
    tree::Tree,
};
pub enum BranchingNodeStatus {
    OnlyVerifierOnChild,
    OnlyVerifierOffChild,
}
// pub struct BranchingNode {
//     pub node_id: usize,
//     pub status: BranchingNodeStatus,
// }

pub fn determine_branching_node(tree: &Tree, rng: &mut impl rand::Rng) -> Option<usize> {
    if tree.leaf_node_ids.len() >= MAX_NUM_TRAJECTORIES {
        println!(
            "Max num trajectories {} reached, finalizing rollout.",
            MAX_NUM_TRAJECTORIES
        );
        return None;
    }

    let mut node_weights = vec![0.0_f64; tree.nodes.len()];
    let mut trajectory_lengths: Vec<usize> = Vec::new();
    for &trajectory_leaf_node_id in &tree.leaf_node_ids {
        let mut trajectory_node_ids_from_leaf_to_root: Vec<usize> = Vec::new();
        let mut cursor = Some(trajectory_leaf_node_id);
        while let Some(node_id) = cursor {
            trajectory_node_ids_from_leaf_to_root.push(node_id);
            let node = tree
                .nodes
                .get(node_id)
                .expect("Trajectory traversal node_id must exist");
            assert_eq!(
                node.node_id, node_id,
                "Node index must equal node_id during trajectory traversal"
            );
            cursor = node.parent_id;
        }
        trajectory_node_ids_from_leaf_to_root.reverse();
        trajectory_lengths.push(trajectory_node_ids_from_leaf_to_root.len());
        if trajectory_node_ids_from_leaf_to_root.len() < 2 {
            println!(
                "Trajectory with leaf node id {} has length less than 2, skipping it for branching node selection.",
                trajectory_leaf_node_id
            );
            continue;
        }
        let per_node_weight = 1.0 / (trajectory_node_ids_from_leaf_to_root.len() - 1) as f64;
        let non_leaf_node_ids = &trajectory_node_ids_from_leaf_to_root
            [..trajectory_node_ids_from_leaf_to_root.len() - 1];
        for &node_id in non_leaf_node_ids {
            node_weights[node_id] += per_node_weight;
        }
    }
    assert_eq!(
        trajectory_lengths.len(),
        tree.leaf_node_ids.len(),
        "Each leaf trajectory should contribute one trajectory length"
    );

    let mut candidate_node_ids: Vec<usize> = Vec::new();
    let mut candidate_weights: Vec<f64> = Vec::new();
    for node in &tree.nodes {
        let weight = node_weights[node.node_id];
        if weight > 0.0 {
            candidate_node_ids.push(node.node_id);
            candidate_weights.push(weight);
        }
    }
    // println!(
    //     "Found {} candidate branching nodes with weights: {:?}",
    //     candidate_node_ids.len(),
    //     candidate_weights
    // );
    while !candidate_node_ids.is_empty() {
        let weighted_index = WeightedIndex::new(&candidate_weights)
            .expect("WeightedIndex construction should succeed with positive candidate weights");
        let sampled_candidate_index = weighted_index.sample(rng);
        let selected_node_id = candidate_node_ids[sampled_candidate_index];
        let selected_node = tree
            .nodes
            .get(selected_node_id)
            .expect("Selected branching node must exist");
        assert_eq!(
            selected_node.node_id, selected_node_id,
            "Node index must equal node_id for selected branching node"
        );
        // let has_verifier_on_child = selected_node.verifier_on_child_id.is_some();
        // let has_verifier_off_child = selected_node.verifier_off_child_id.is_some();
        let mut has_verifier_on_child = false;
        let mut has_verifier_off_child = false;
        for &child_id in &selected_node.child_ids {
            let Some(child_id) = child_id else {
                continue;
            };
            let child_node = tree
                .nodes
                .get(child_id)
                .expect("Child node of selected branching node must exist");
            match child_node.step.node_type() {
                NodeType::VerifierOn
                | NodeType::VerifierOnAndChangePlan
                | NodeType::VerifierOnAndOverwriteLastStep => {
                    assert!(
                        !has_verifier_on_child,
                        "A branching node should not have more than one verifier on child"
                    );
                    has_verifier_on_child = true;
                }
                NodeType::VerifierOff => {
                    assert!(
                        !has_verifier_off_child,
                        "A branching node should not have more than one verifier off child"
                    );
                    has_verifier_off_child = true;
                }
            }
        }
        match (has_verifier_on_child, has_verifier_off_child) {
            (true, true) => {
                // this node has both verifier on and off children, we skip it and resample
                candidate_node_ids.swap_remove(sampled_candidate_index);
                candidate_weights.swap_remove(sampled_candidate_index);
                continue;
            }
            (true, false) | (false, true) => {
                // return Some(BranchingNode {
                //     node_id: selected_node_id,
                //     status: BranchingNodeStatus::OnlyVerifierOnChild,
                // });
                return Some(selected_node_id);
            }
            // (false, true) => {
            //     return Some(BranchingNode {
            //         node_id: selected_node_id,
            //         status: BranchingNodeStatus::OnlyVerifierOffChild,
            //     });
            // }
            (false, false) => {
                panic!(
                    "Selected branching node must have at least one child with verifier on or off"
                )
            }
        }
    }
    println!("No valid branching node found, finalizing rollout.");
    None
}
