use crate::{agent::tree_schema::CompletedTree, training_set::training_set_formatted::TrainingSampleFormatted};

// The prompt is collected when the TrajectoryStatus enters "PlannerWorkingOnStep" (after the action TrajectoryAction::PlannerMakeOrChangePlan)
// The response content is composed of the following actions after that point:
// TrajectoryAction::PlannerReasoning
// TrajectoryAction::PlannerToolCall
// TrajectoryAction::ToolCallResponse
// For TrajectoryAction::ToolCallResponse specifically, it needs to be wrapped in <__tool_response_start_> and <_tool_response_end_> tags in the response content.
pub fn generate_sample_formatted_from_tree_node(
    tree: &CompletedTree,
    node_id: usize,
) -> TrainingSampleFormatted {
    todo!()
}
