use std::collections::BTreeSet;

use crate::{
    advantage_composition::AdvantageCompositionPerTree,
    agent::{
        trajectory_action::{TrajectoryAction, TrajectoryActionLog},
        trajectory_state::TrajectoryState,
        tree_schema::CompletedTree,
    },
    apply_vllm_model_chat_template::apply_vllm_model_chat_template,
    constants::{
        FIXED_ADVANTAGE_WEIGHT_CONTRIBUTION, FIXED_ADVANTAGE_WEIGHT_STEP_QUALITY,
        FIXED_ADVANTAGE_WEIGHT_TRAJECTORY,
    },
    direct_answer::generate_raw_answers::LlmModel,
    status_prompts::universal_prompt::get_prompt_according_to_session_status,
    training_set::training_set_formatted::TrainingSampleFormatted,
};

// The prompt is collected when the TrajectoryStatus enters "PlannerWorkingOnStep" (after the action TrajectoryAction::PlannerMakeOrChangePlan)
// The formatted content uses mask tags:
// - <__start_mask__>
// - <__end_mask_with_eos__>
// Only assistant response content is wrapped by mask tags, while prompt and tool responses stay
// outside masked regions. The full content ends with <__end_mask_with_eos__>.
// The response content is composed of the following actions after that point:
// TrajectoryAction::PlannerReasoning
// TrajectoryAction::PlannerToolCall
// TrajectoryAction::ToolCallResponse
// For TrajectoryAction::ToolCallResponse specifically, it is emitted outside masked regions.
pub fn generate_sample_formatted_from_tree_node(
    tree: &CompletedTree,
    advantage_composition: &AdvantageCompositionPerTree,
    node_id: usize,
    model: LlmModel,
) -> TrainingSampleFormatted {
    assert!(model.is_qwen(), "Training sample formatting currently requires a Qwen model");
    assert_eq!(
        tree.id, tree.trajectory.question_id,
        "CompletedTree.id must equal Tree.question_id"
    );
    assert_eq!(
        advantage_composition.question_id, tree.id,
        "AdvantageCompositionPerTree.question_id must match CompletedTree.id"
    );
    assert!(
        node_id < tree.trajectory.nodes.len(),
        "node_id must be in-bounds for trajectory nodes"
    );

    let target_node = &tree.trajectory.nodes[node_id];
    assert_eq!(target_node.node_id, node_id, "Node index must equal node_id");

    let mut path_ids_from_target_to_root: Vec<usize> = Vec::new();
    let mut seen: BTreeSet<usize> = BTreeSet::new();
    let mut cursor = Some(node_id);
    while let Some(current_id) = cursor {
        assert!(
            seen.insert(current_id),
            "Tree path to node_id must not contain cycles"
        );
        assert!(
            current_id < tree.trajectory.nodes.len(),
            "Path traversal node_id must be in-bounds"
        );
        let node = &tree.trajectory.nodes[current_id];
        assert_eq!(node.node_id, current_id, "Node index must equal node_id");
        path_ids_from_target_to_root.push(current_id);
        cursor = node.parent_id;
    }
    assert!(
        !path_ids_from_target_to_root.is_empty(),
        "Path to root must be non-empty"
    );
    path_ids_from_target_to_root.reverse();

    let mut actions_until_step_start: Vec<TrajectoryAction> = Vec::new();
    for path_node_id in &path_ids_from_target_to_root {
        let node = &tree.trajectory.nodes[*path_node_id];
        if *path_node_id != node_id {
            actions_until_step_start.extend(node.step.action_log.iter().cloned());
            continue;
        }

        let mut found_make_or_change_plan = false;
        for action in &node.step.action_log {
            actions_until_step_start.push(action.clone());
            if matches!(action, TrajectoryAction::PlannerMakeOrChangePlan(_)) {
                found_make_or_change_plan = true;
                break;
            }
        }
        assert!(
            found_make_or_change_plan,
            "Target node action log must contain PlannerMakeOrChangePlan"
        );
    }

    let session_state = TrajectoryState::from_session_log(
        tree.question.clone(),
        TrajectoryActionLog(actions_until_step_start),
        &tree.trajectory,
    );
    let (prompt_before_assistant, prompt_after_assistant) =
        get_prompt_according_to_session_status(&session_state);

    let mut response_content = String::new();
    let mut in_mask_segment = false;
    let mut collecting = false;
    for action in &target_node.step.action_log {
        if collecting {
            match action {
                TrajectoryAction::PlannerReasoning { reasoning } => {
                    if !in_mask_segment {
                        response_content.push_str("<__start_mask__>");
                        in_mask_segment = true;
                    }
                    response_content.push_str(reasoning);
                }
                TrajectoryAction::PlannerToolCall(tool_call) => {
                    if !in_mask_segment {
                        response_content.push_str("<__start_mask__>");
                        in_mask_segment = true;
                    }
                    response_content.push_str(tool_call);
                }
                TrajectoryAction::ToolCallResponse(tool_response) => {
                    if in_mask_segment {
                        response_content.push_str("<__end_mask_with_eos__>");
                        in_mask_segment = false;
                    }
                    response_content.push_str(&tool_response.to_raw_content());
                }
                _ => {}
            }
            continue;
        }

        if matches!(action, TrajectoryAction::PlannerMakeOrChangePlan(_)) {
            collecting = true;
        }
    }
    if !in_mask_segment {
        response_content.push_str("<__start_mask__>");
        in_mask_segment = true;
    }
    if in_mask_segment {
        response_content.push_str("<__end_mask_with_eos__>");
    }

    let mut planner_chat_template_prompt =
        apply_vllm_model_chat_template(model, &prompt_before_assistant, false);
    planner_chat_template_prompt.push_str(&prompt_after_assistant);

    let node_advantage = advantage_composition
        .per_node
        .iter()
        .find(|per_node| per_node.node_id == node_id)
        .expect("AdvantageCompositionPerTree must include an entry for target node_id");
    let step_quality_sum = node_advantage.step_quality_tool_advantage_normalized
        + node_advantage.step_quality_complete_advantage_normalized
        + node_advantage.step_quality_focused_advantage_normalized;
    let per_step_advantage = node_advantage.contribution_mean_div_var_normalized
        * FIXED_ADVANTAGE_WEIGHT_CONTRIBUTION
        + node_advantage.trajectory_advantage_normalized * FIXED_ADVANTAGE_WEIGHT_TRAJECTORY
        + step_quality_sum * FIXED_ADVANTAGE_WEIGHT_STEP_QUALITY;

    TrainingSampleFormatted {
        question_id: tree.id,
        node_id,
        content_formatted: format!("{}{}", planner_chat_template_prompt, response_content),
        advantage: per_step_advantage,
    }
}
