use rand::RngExt;
use reqwest::Client;

use crate::agent::context_length_exceeded::{
    context_length_exceeded_result, is_context_length_exceeded_response,
};
use crate::agent::response_processing::{
    ChosenModeDecision, determine_chosen_mode, parse_compactor_response,
    parse_verifier_comment_response, split_reasoning_and_tool_call,
};
use crate::agent::tool_call_execution::{MAX_NUM_TRAJECTORIES, execute_planner_tool_call};
use crate::agent::trajectory_action_types::{
    FinalAnswer, MakeOrChangePlan, NextStepDecision, ToolResponse, VerifierAndModeSummary,
};
use crate::agent::trajectory_state::TrajectoryState;
use crate::agent::trajectory_status::TrajectoryStatus;
use crate::agent::tree::{CorrectnessJudgment, Tree};
use crate::agent::tree_action::TreeAction;
use crate::status_prompts::universal_prompt::get_prompt_according_to_session_status;
use crate::{
    agent::trajectory_action::TrajectoryAction,
    call_llm::call_llm_with_prefix,
    constants::{IDENTICAL_PYTHON_ERROR_ABORT_MESSAGE, REPETITION_ABORT_MESSAGE},
    direct_answer::{generate_raw_answers::LlmModel, judge_answers::judge_answer_task},
};

// we increased the repetition times to 5, there might be code that hasn't reflected this change.
pub fn detect_repetition_five_times(response: &str) -> bool {
    let min_subsequence_length = 50; // minimum length of the repeated subsequence to avoid false positive from short common phrases

    let bytes = response.as_bytes();
    let n = bytes.len();
    if n < min_subsequence_length * 5 {
        return false;
    }

    // Rolling hash over raw bytes. We still verify by byte comparison when hashes match.
    let base: u64 = 1_000_003;
    let mut pow = vec![0_u64; n + 1];
    let mut prefix = vec![0_u64; n + 1];
    pow[0] = 1;
    for i in 0..n {
        pow[i + 1] = pow[i].wrapping_mul(base);
        prefix[i + 1] = prefix[i]
            .wrapping_mul(base)
            .wrapping_add((bytes[i] as u64) + 1);
    }

    let hash = |start: usize, len: usize| -> u64 {
        prefix[start + len].wrapping_sub(prefix[start].wrapping_mul(pow[len]))
    };

    for len in min_subsequence_length..=(n / 5) {
        for start in 0..=(n - 5 * len) {
            let h1 = hash(start, len);
            let h2 = hash(start + len, len);
            let h3 = hash(start + 2 * len, len);
            let h4 = hash(start + 3 * len, len);
            let h5 = hash(start + 4 * len, len);
            if h1 != h2 || h1 != h3 || h1 != h4 || h1 != h5 {
                continue;
            }

            let s1 = &bytes[start..start + len];
            let s2 = &bytes[start + len..start + 2 * len];
            let s3 = &bytes[start + 2 * len..start + 3 * len];
            let s4 = &bytes[start + 3 * len..start + 4 * len];
            let s5 = &bytes[start + 4 * len..start + 5 * len];
            if s1 == s2 && s1 == s3 && s1 == s4 && s1 == s5 {
                return true;
            }
        }
    }

    false
}

pub async fn produce_actions_from_state(
    // session_state: &TrajectoryState<'_>,
    tree: &Tree,
    client: Client,
    model: LlmModel,
    take_over_mode_decision: bool,
    rng: &mut impl rand::Rng,
) -> Vec<TreeAction> {
    let session_state = TrajectoryState::from_tree(tree);
    // let tree = &session_state.source_tree;
    let question_id = tree.question_id;
    match &session_state.status {
        TrajectoryStatus::StepEnded => {
            let mut operations: Vec<TreeAction> = Vec::new();

            let (node_id, parent_id) = if tree.current_node_id.is_none() {
                assert!(
                    tree.root_node_id.is_none() && tree.nodes.is_empty() && tree.next_node_id == 0,
                    "Tree without current node must be an uninitialized empty tree"
                );
                (0, None)
            } else {
                let parent_node_id = tree
                    .current_node_id
                    .expect("VerifierCommenting requires current_node_id");
                let next_node_id = tree.next_node_id;
                (next_node_id, Some(parent_node_id))
            };
            operations.push(TreeAction::CreateNode {
                question_id,
                node_id,
                parent_id,
            });
            operations.push(TreeAction::SetCurrentNode {
                question_id,
                node_id,
            });
            operations.push(TreeAction::AddTrajectoryAction {
                question_id,
                action: TrajectoryAction::StartNewStep, // set the trajectory status to verifier commenting, finalize the old step node
            });
            operations
        }
        TrajectoryStatus::VerifierCommenting => {
            let parent_id = session_state
                .source_tree
                .get_current_node()
                .expect("VerifierCommenting current node must exist")
                .parent_id;

            let verifier_on = if let Some(parent_node_id) = parent_id {
                let parent_node = tree.get_node_by_id(parent_node_id);
                assert!(
                    parent_node.child_ids[0].is_none() || parent_node.child_ids[1].is_none(),
                    "VerifierCommenting parent already has both children assigned"
                );
                let existing_verifier_on = parent_node.child_ids.iter().find_map(|id| {
                    let Some(id) = id else {
                        return None;
                    };
                    let other_child = tree.get_node_by_id(*id);
                    let verifier_and_mode = other_child.step.verifier_and_mode_summary();
                    match &verifier_and_mode {
                        VerifierAndModeSummary::VerifierOff => Some(false),
                        _ => Some(true),
                    }
                });
                existing_verifier_on
                    .and_then(|v| Some(!v))
                    .unwrap_or_else(|| rng.random::<f32>() < 0.5)
            } else {
                // if there is no parent node, then verifier should be off because there is not last step
                false
            };
            let actions = if verifier_on {
                assert!(!session_state.prev_steps.is_empty());
                let (prompt_before_assistant, prompt_after_assistant) =
                    get_prompt_according_to_session_status(&session_state);
                assert_eq!(
                    prompt_after_assistant,
                    String::new(),
                    "Verifier commenting should not have prompt after assistant"
                );
                let response = call_llm_with_prefix(
                    client.clone(),
                    prompt_before_assistant,
                    prompt_after_assistant,
                    model,
                )
                .await;
                if is_context_length_exceeded_response(&response) {
                    context_length_exceeded_result(question_id)
                } else {
                    let action = TreeAction::AddTrajectoryAction {
                        question_id,
                        action: TrajectoryAction::VerifierComment(Some(
                            parse_verifier_comment_response(response.clone()),
                        )),
                    };
                    vec![action]
                }
            } else {
                // if verifier is off, we still want to add a TrajectoryAction to record the verifier comment is off for the current step, which will be used for determining the mode in the next step
                let action = TreeAction::AddTrajectoryAction {
                    question_id,
                    action: TrajectoryAction::VerifierComment(None),
                };
                vec![action]
            };
            actions
        }
        TrajectoryStatus::PlannerMakingOrChangingPlan {
            planner_chosen_mode,
            verifier_comment: _,
        } => {
            let (prompt_before_assistant, prompt_after_assistant) =
                get_prompt_according_to_session_status(&session_state);
            let response = call_llm_with_prefix(
                client.clone(),
                prompt_before_assistant,
                prompt_after_assistant,
                model,
            )
            .await;
            if is_context_length_exceeded_response(&response) {
                context_length_exceeded_result(question_id)
            } else {
                let plan_content = match planner_chosen_mode {
                    NextStepDecision::ChangePlan(reason) => Some(MakeOrChangePlan::ChangePlan {
                        plan: response,
                        prev_failed_reason: reason.clone(),
                    }),
                    NextStepDecision::Continue | NextStepDecision::OverwriteLastStep(_) => {
                        Some(MakeOrChangePlan::MakePlan(response))
                    }
                }; // we change to not require the plan to be in a markdown code block
                vec![TreeAction::AddTrajectoryAction {
                    question_id,
                    action: TrajectoryAction::PlannerMakeOrChangePlan(plan_content),
                }]
            }
        }
        TrajectoryStatus::PlannerChoosingMode {
            verifier_comment: _,
        } => {
            match determine_chosen_mode(
                &session_state,
                client.clone(),
                model,
                take_over_mode_decision,
                rng,
            )
            .await
            {
                ChosenModeDecision::ContextLengthExceeded => {
                    context_length_exceeded_result(question_id)
                }
                ChosenModeDecision::Chosen(chosen_mode) => vec![TreeAction::AddTrajectoryAction {
                    question_id,
                    action: TrajectoryAction::PlannerDecideNextStep(chosen_mode),
                }],
            }
        }
        TrajectoryStatus::PlannerWorkingOnStep {
            planner_chosen_mode: _,
            verifier_comment: _,
            step_content_raw,
        } => {
            let (prompt_before_assistant, prompt_after_assistant) =
                get_prompt_according_to_session_status(&session_state);
            let mut response = call_llm_with_prefix(
                client.clone(),
                prompt_before_assistant,
                prompt_after_assistant,
                model,
            )
            .await;
            if is_context_length_exceeded_response(&response) {
                context_length_exceeded_result(question_id)
            } else {
                if response.trim().is_empty() {
                    response = "<end_step>".to_string(); // if the model does not output anything, we treat it as if it outputs <end_step> to prevent getting stuck
                }
                let response_is_empty =
                    response.trim() == "<end_step>" && step_content_raw.trim().is_empty();
                if response_is_empty {
                    println!(
                        "[Warning]: model tries to end the step without providing any content for the step."
                    );
                }
                let (reasoning, tool_call, tool_wait_violation) =
                    split_reasoning_and_tool_call(response.clone(), model);
                let mut push_end_step = false;
                let mut has_terminal_intervention = false;
                let mut operations: Vec<TreeAction> = Vec::new();
                if tool_wait_violation {
                    operations.push(TreeAction::ToolWaitViolation { question_id });
                }
                if let Some(reasoning) = reasoning {
                    if reasoning.contains("<end_step>") {
                        push_end_step = true;
                    }
                    operations.push(TreeAction::AddTrajectoryAction {
                        question_id,
                        action: TrajectoryAction::PlannerReasoning { reasoning },
                    });
                }
                if let Some(tool_call) = tool_call {
                    let tool_response = execute_planner_tool_call(&tool_call).await;
                    let previous_python_error =
                        session_state.current_step_last_python_error.clone();
                    operations.push(TreeAction::AddTrajectoryAction {
                        question_id,
                        action: TrajectoryAction::PlannerToolCall(tool_call),
                    });
                    if let ToolResponse::PythonError(current_python_error) = &tool_response {
                        if previous_python_error.is_some()
                            && Some(current_python_error.clone()) == previous_python_error
                        {
                            println!(
                                "[Warning]: Identical python tool error detected. Aborting current step."
                            );
                            operations.push(TreeAction::AddTrajectoryAction {
                                question_id,
                                action: TrajectoryAction::ToolCallResponse(tool_response),
                            });
                            operations.push(TreeAction::AddTrajectoryAction {
                                question_id,
                                // action: RolloutAction::ToolCallResponse(
                                //     ToolResponse::Intervention(
                                //         IDENTICAL_PYTHON_ERROR_ABORT_MESSAGE.to_string(),
                                //     ),
                                // ),
                                action: TrajectoryAction::SystemInterrupt(
                                    IDENTICAL_PYTHON_ERROR_ABORT_MESSAGE.to_string(),
                                ),
                            });
                            has_terminal_intervention = true;
                        } else {
                            operations.push(TreeAction::AddTrajectoryAction {
                                question_id,
                                action: TrajectoryAction::ToolCallResponse(tool_response),
                            });
                        }
                    } else {
                        operations.push(TreeAction::AddTrajectoryAction {
                            question_id,
                            action: TrajectoryAction::ToolCallResponse(tool_response),
                        });
                    }
                }

                if response_is_empty {
                    operations.push(TreeAction::AddTrajectoryAction {
                        question_id,
                        // action: RolloutAction::ToolCallResponse(ToolResponse::Intervention(
                        //     SUBMIT_ANSWER_HINT.to_string(),
                        // )),
                        action: TrajectoryAction::ToolCallResponse(ToolResponse::EmptyMessageHint),
                    });
                }
                let num_additional_actions_allowed =
                    session_state.num_additional_actions_allowed_in_current_step();
                if operations.len() > num_additional_actions_allowed {
                    println!(
                        "[Warning] Number of actions in the current step {} exceeds the limit {}. Only the first {} actions will be applied.",
                        operations.len(),
                        num_additional_actions_allowed,
                        num_additional_actions_allowed
                    );
                    operations.truncate(num_additional_actions_allowed);
                }
                // detect repetition
                let found_repetition_three_times = detect_repetition_five_times(&response);
                if found_repetition_three_times {
                    println!(
                        "[Warning] Detected repetition of the same response at least three times. This may indicate that the model is stuck in a loop. Response: {}",
                        response
                    );
                    operations.push(TreeAction::AddTrajectoryAction {
                        question_id,
                        action: TrajectoryAction::SystemInterrupt(
                            REPETITION_ABORT_MESSAGE.to_string(),
                        ),
                    });
                    has_terminal_intervention = true;
                }

                let current_step_full = operations.len() == num_additional_actions_allowed;
                if (push_end_step && !response_is_empty) || current_step_full {
                    assert!(
                        !has_terminal_intervention,
                        "PlannerEndStep should not be emitted after terminal intervention"
                    );
                    operations.push(TreeAction::AddTrajectoryAction {
                        question_id,
                        action: TrajectoryAction::PlannerEndStep,
                    });
                }
                operations
            }
        }
        TrajectoryStatus::CompactorCompactingStep {
            planner_chosen_mode: _,
            step_content_raw: _,
        } => {
            let (prompt_before_assistant, prompt_after_assistant) =
                get_prompt_according_to_session_status(&session_state);
            let response = call_llm_with_prefix(
                client.clone(),
                prompt_before_assistant,
                prompt_after_assistant,
                model,
            )
            .await;
            if is_context_length_exceeded_response(&response) {
                context_length_exceeded_result(question_id)
            } else {
                let (step_content_compacted, step_quality) = parse_compactor_response(response);
                vec![TreeAction::AddTrajectoryAction {
                    question_id,
                    action: TrajectoryAction::CompactorCompactStep {
                        step_content_compacted,
                        step_quality,
                    },
                }]
            }
        }
        TrajectoryStatus::PlannerUpdatingPlan {
            planner_chosen_mode: _,
            step_content_raw: _,
            step_content_compacted: _,
        } => {
            let (prompt_before_assistant, prompt_after_assistant) =
                get_prompt_according_to_session_status(&session_state);
            let response = call_llm_with_prefix(
                client.clone(),
                prompt_before_assistant,
                prompt_after_assistant,
                model,
            )
            .await;
            if is_context_length_exceeded_response(&response) {
                context_length_exceeded_result(question_id)
            } else {
                let updated_plan_content = response; // we change to not require the updated plan to be in a markdown code block
                vec![TreeAction::AddTrajectoryAction {
                    question_id,
                    action: TrajectoryAction::PlannerUpdatePlan(Some(updated_plan_content)),
                }]
            }
        }
        TrajectoryStatus::SessionEnded { final_answer } => {
            let mut actions = Vec::new();
            let display_final_answer = match final_answer {
                FinalAnswer::ModelProvided(ans) => ans.clone(),
                FinalAnswer::Failure(reason) => format!("Failure: {}", reason),
            };
            println!(
                "[rollout finished] question index: {}, total actual rounds: {}, final answer: {}, correct answer: {}",
                tree.question_id,
                session_state.total_actions,
                display_final_answer,
                tree.reference_answer,
            );
            let leaf_node_id = tree
                .current_node_id
                .expect("WorkingOnTrajectory should always have current_node_id when ending");
            assert!(
                !tree.leaf_node_ids.contains(&leaf_node_id),
                "The leaf node for the trajectory should not have been registered yet"
            );
            assert!(
                !tree.leaf_node_judgments.contains_key(&leaf_node_id),
                "The leaf node for the trajectory should not have been judged yet"
            );
            let register_leaf_action = TreeAction::RegisterLeaf {
                question_id: tree.question_id,
                node_id: leaf_node_id,
            };
            actions.push(register_leaf_action);
            // let model_answer = extract_leaf_model_answer(tree, leaf_node_id);
            let is_correct = match final_answer {
                FinalAnswer::ModelProvided(final_answer) => {
                    judge_answer_task(
                        tree.question_id,
                        final_answer.clone(),
                        tree.reference_answer.clone(),
                        tree.question.clone(),
                        client.clone(),
                    )
                    .await
                }
                FinalAnswer::Failure(_) => false, // if the model fails to provide a final answer, we treat it as an incorrect answer
            };
            println!(
                "[judgment] question id: {}, trajectory: {}/{}, model answer: {}, reference answer: {}, is correct: {}",
                tree.question_id,
                tree.leaf_node_ids.len(),
                MAX_NUM_TRAJECTORIES,
                display_final_answer,
                tree.reference_answer,
                is_correct
            );
            let judge_leaf_correctness_event = TreeAction::JudgeLeafCorrectness {
                question_id: tree.question_id,
                node_id: leaf_node_id,
                correctness_judgment: CorrectnessJudgment {
                    model_answer: final_answer.clone(),
                    correct_answer: tree.reference_answer.clone(),
                    is_correct,
                },
            };
            actions.push(judge_leaf_correctness_event);
            actions
        }
    }
}
