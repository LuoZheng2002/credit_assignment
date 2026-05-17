use rand::RngExt;
use reqwest::Client;

use crate::agent::branching_node_selection::determine_branching_node;
use crate::agent::context_length_exceeded::{
    context_length_exceeded_result, is_context_length_exceeded_response,
};
use crate::agent::response_processing::{
    determine_chosen_mode, parse_compactor_response, parse_verifier_comment_response,
    split_reasoning_and_tool_call,
};
use crate::agent::tool_call_execution::execute_planner_tool_call;
use crate::agent::trajectory_action_types::{
    FinalAnswer, MakeOrChangePlan, NextStepDecision, NodeType, ToolResponse,
};
use crate::agent::trajectory_state::TrajectoryState;
use crate::agent::trajectory_status::TrajectoryStatus;
use crate::agent::tree::{CorrectnessJudgment, Tree};
use crate::agent::tree_action::TreeAction;
use crate::call_llm::{LlmCallable, call_llm_chat_completions, call_llm_with_prefix};
use crate::llm_models::LlmModelMarker;
use crate::status_prompts::universal_prompt::get_prompt_according_to_session_status;
use crate::util::extract_boxed_content;
use crate::worker_message_tx::log_key_value_pair;
use crate::{
    agent::trajectory_action::TrajectoryAction,
    constants::{IDENTICAL_PYTHON_ERROR_ABORT_MESSAGE, REPETITION_ABORT_MESSAGE},
    llm_model::LlmModel,
};

pub async fn judge_answer_task(
    question_id: usize,
    model_answer: String,
    correct_answer: String,
    question: String,
    client: Client,
) -> bool {
    let prompt = format!(
        "You are an answer checker that checks a model's answer against the reference answer. Judge if the model's answer is equivalent to the reference answer. \
If the model's answer contains units but the reference answer does not, treat them as equivalent if the numerical values are the same. \n\
The question is: \"{}\". \
The model's answer is: \"{}\", and the correct answer is: \"{}\". Return only 'correct' or 'incorrect'.",
        question, model_answer, correct_answer
    );
    let evaluation = call_llm_chat_completions(client, prompt, LlmModel::Gpt4o, false)
        .await
        .trim()
        .to_lowercase();
    match evaluation.as_str() {
        "correct" => true,
        "incorrect" => false,
        _ => panic!(
            "Unexpected evaluation result for question {}: {}",
            question_id, evaluation
        ),
    }
}

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

pub async fn produce_actions_from_state<M: LlmModelMarker, C: LlmCallable<M>>(
    // session_state: &TrajectoryState<'_>,
    tree: &Tree,
    llm_callable: &C,
    client: Client,
    rng: &mut impl rand::Rng,
) -> Vec<TreeAction> {
    let session_state = TrajectoryState::from_tree(tree);
    // let tree = &session_state.source_tree;
    let question_id = tree.question_id;
    match &session_state.status {
        // adding an empty variant does not seem to work because
        // we need to start with empty for the tree to create the first node
        // but at the same time we cannot transition to verifier commenting without taking a "start new step" action
        // therefore, there is not good way to manipulate the tree based only on the trajectory's status
        // but we do want to recover trajectory state and status based on the trajectory action log stored in the tree
        // we may initialize both the tree and the trajectory state
        // TrajectoryStatus::Empty => {
        //     assert!(
        //         tree.root_node_id.is_none() && tree.nodes.is_empty() && tree.next_node_id == 0,
        //         "Tree without current node must be an uninitialized empty tree"
        //     );
        //     vec![TreeAction::CreateAndMoveToNode {
        //         question_id,
        //         parent_id: None,
        //     }]
        // }
        TrajectoryStatus::StepEnded => {
            let mut actions: Vec<TreeAction> = Vec::new();
            assert!(tree.current_node_id.is_some());

            let parent_id = tree
                .current_node_id
                .expect("Step ended requires current_node_id");
            actions.push(TreeAction::AddTrajectoryAction {
                question_id,
                action: TrajectoryAction::StartNewStep, // set the trajectory status to verifier commenting, finalize the old step node
            });
            actions.push(TreeAction::CreateAndMoveToNode {
                question_id,
                parent_id: Some(parent_id),
            });
            actions
        }
        TrajectoryStatus::VerifierCommenting => {
            let current_node = session_state
                .source_tree
                .get_current_node()
                .expect("VerifierCommenting current node must exist");
            let current_node_id = current_node.node_id;
            let verifier_on = match current_node.parent_id {
                None => false,
                Some(parent_id) => {
                    let parent_node = tree.get_node_by_id(parent_id);
                    let sibling_id = parent_node
                        .child_ids
                        .iter()
                        .copied()
                        .flatten()
                        .find(|id| *id != current_node_id);
                    let existing_verifier_on = sibling_id.map(|id| {
                        let sibling_node = tree.get_node_by_id(id);
                        let sibling_node_type = sibling_node.step.node_type();
                        match sibling_node_type {
                            NodeType::VerifierOff => false,
                            _ => true,
                        }
                    });
                    existing_verifier_on
                        .map(|v| !v)
                        .unwrap_or_else(|| rng.random::<f32>() < 0.5)
                }
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
                let response = call_llm_with_prefix::<M, C>(
                    llm_callable,
                    prompt_before_assistant,
                    prompt_after_assistant,
                )
                .await;
                if is_context_length_exceeded_response(&response) {
                    context_length_exceeded_result(
                        question_id,
                        session_state.final_answer.is_some(),
                    )
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
                // if verifier is off, we still want to add a TrajectoryAction to record the verifier comment is off for the current step, which will be used for determining the node type in the next step
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
            let needs_to_make_or_change_plan = match planner_chosen_mode {
                NextStepDecision::Continue => session_state.current_plan.is_none(),
                NextStepDecision::OverwriteLastStep(_) => {
                    assert!(
                        session_state.current_plan.is_some(),
                        "OverwriteLastStep requires existing plan to overwrite"
                    );
                    false
                }
                NextStepDecision::ChangePlan(_) => true,
            };
            if needs_to_make_or_change_plan {
                let (prompt_before_assistant, prompt_after_assistant) =
                    get_prompt_according_to_session_status(&session_state);
                let response = call_llm_with_prefix::<M, C>(
                    llm_callable,
                    prompt_before_assistant,
                    prompt_after_assistant,
                )
                .await;
                if is_context_length_exceeded_response(&response) {
                    context_length_exceeded_result(
                        question_id,
                        session_state.final_answer.is_some(),
                    )
                } else {
                    let plan_content = match planner_chosen_mode {
                        NextStepDecision::ChangePlan(reason) => {
                            Some(MakeOrChangePlan::ChangePlan {
                                plan: response,
                                prev_failed_reason: reason.clone(),
                            })
                        }
                        NextStepDecision::Continue => Some(MakeOrChangePlan::MakePlan(response)),
                        NextStepDecision::OverwriteLastStep(_) => unreachable!(),
                    }; // we change to not require the plan to be in a markdown code block
                    vec![TreeAction::AddTrajectoryAction {
                        question_id,
                        action: TrajectoryAction::PlannerMakeOrChangePlan(plan_content),
                    }]
                }
            } else {
                vec![TreeAction::AddTrajectoryAction {
                    question_id,
                    action: TrajectoryAction::PlannerMakeOrChangePlan(None),
                }]
            }
        }
        TrajectoryStatus::PlannerChoosingMode {
            verifier_comment: _,
        } => vec![determine_chosen_mode(&session_state, rng).await],
        TrajectoryStatus::PlannerWorkingOnStep {
            planner_chosen_mode: _,
            verifier_comment: _,
            step_content_raw,
        } => {
            let (prompt_before_assistant, prompt_after_assistant) =
                get_prompt_according_to_session_status(&session_state);
            let mut response = call_llm_with_prefix::<M, C>(
                llm_callable,
                prompt_before_assistant,
                prompt_after_assistant,
            )
            .await;
            if is_context_length_exceeded_response(&response) {
                context_length_exceeded_result(question_id, session_state.final_answer.is_some())
            } else {
                let mut actions: Vec<TreeAction> = Vec::new();
                if session_state.final_answer.is_none()
                    && let Some(boxed_content) = extract_boxed_content(&response)
                {
                    actions.push(TreeAction::AddTrajectoryAction {
                        question_id,
                        action: TrajectoryAction::SubmitFinalAnswer(FinalAnswer::ModelProvided(
                            boxed_content,
                        )),
                    });
                }
                if response.trim().is_empty() {
                    response = "<end_step>".to_string(); // if the model does not output anything, we treat it as if it outputs <end_step> to prevent getting stuck
                }
                let response_is_empty =
                    response.trim() == "<end_step>" && step_content_raw.trim().is_empty();
                if response_is_empty {
                    log_key_value_pair(
                        "warning".into(),
                        "Model tries to end the step without providing any content for the step."
                            .into(),
                    );
                }
                let (reasoning, tool_call, tool_wait_violation) =
                    split_reasoning_and_tool_call(response.clone());
                let mut push_end_step = false;
                let mut has_step_terminate_intervention = false;

                if tool_wait_violation {
                    actions.push(TreeAction::ToolWaitViolation { question_id });
                }
                if let Some(reasoning) = reasoning {
                    if reasoning.contains("<end_step>") {
                        push_end_step = true;
                    }
                    actions.push(TreeAction::AddTrajectoryAction {
                        question_id,
                        action: TrajectoryAction::PlannerReasoning { reasoning },
                    });
                }
                if let Some(tool_call) = tool_call {
                    let tool_response = execute_planner_tool_call(&tool_call).await;
                    let previous_python_error =
                        session_state.current_step_last_python_error.clone();
                    actions.push(TreeAction::AddTrajectoryAction {
                        question_id,
                        action: TrajectoryAction::PlannerToolCall(tool_call),
                    });
                    if let ToolResponse::PythonError(current_python_error) = &tool_response {
                        if previous_python_error.is_some()
                            && Some(current_python_error.clone()) == previous_python_error
                        {
                            log_key_value_pair(
                                "warning".into(),
                                "Identical python tool error detected. Aborting current step."
                                    .into(),
                            );
                            actions.push(TreeAction::AddTrajectoryAction {
                                question_id,
                                action: TrajectoryAction::ToolCallResponse(tool_response),
                            });
                            actions.push(TreeAction::AddTrajectoryAction {
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
                            has_step_terminate_intervention = true;
                        } else {
                            actions.push(TreeAction::AddTrajectoryAction {
                                question_id,
                                action: TrajectoryAction::ToolCallResponse(tool_response),
                            });
                        }
                    } else {
                        actions.push(TreeAction::AddTrajectoryAction {
                            question_id,
                            action: TrajectoryAction::ToolCallResponse(tool_response),
                        });
                    }
                }

                if response_is_empty {
                    actions.push(TreeAction::AddTrajectoryAction {
                        question_id,
                        // action: RolloutAction::ToolCallResponse(ToolResponse::Intervention(
                        //     SUBMIT_ANSWER_HINT.to_string(),
                        // )),
                        // action: TrajectoryAction::ToolCallResponse(ToolResponse::EmptyMessageHint),
                        // action: TrajectoryAction::SystemInterrupt("Model tries to end the step at the beginning of a step.".into())
                        action: TrajectoryAction::SubmitFinalAnswer(FinalAnswer::Failure(
                            "Model tries to end the step at the beginning of a step.".into(),
                        )),
                    });
                    has_step_terminate_intervention = true;
                }
                let num_additional_actions_allowed =
                    session_state.num_additional_actions_allowed_in_current_step();
                if actions.len() > num_additional_actions_allowed {
                    log_key_value_pair(
                        "warning".into(),
                        format!(
                            "Number of actions in the current step {} exceeds the limit {}. Only the first {} actions will be applied.",
                            actions.len(),
                            num_additional_actions_allowed,
                            num_additional_actions_allowed
                        ),
                    );
                    actions.truncate(num_additional_actions_allowed);
                }
                // detect repetition
                let found_repetition_three_times = detect_repetition_five_times(&response);
                if found_repetition_three_times {
                    log_key_value_pair(
                            "warning".into(),
                            "Detected repetition of the same response at least five times. This may indicate that the model is stuck in a loop.".into(),
                        );
                    actions.push(TreeAction::AddTrajectoryAction {
                        question_id,
                        action: TrajectoryAction::SystemInterrupt(
                            REPETITION_ABORT_MESSAGE.to_string(),
                        ),
                    });
                    has_step_terminate_intervention = true;
                }

                let current_step_full = actions.len() == num_additional_actions_allowed;
                if ((push_end_step && !response_is_empty) || current_step_full)
                    && !has_step_terminate_intervention
                {
                    // assert!(
                    //     !has_terminal_intervention,
                    //     "PlannerEndStep should not be emitted after terminal intervention"
                    // );
                    actions.push(TreeAction::AddTrajectoryAction {
                        question_id,
                        action: TrajectoryAction::PlannerEndStep,
                    });
                }
                actions
            }
        }
        TrajectoryStatus::CompactorCompactingStep {
            planner_chosen_mode: _,
            step_content_raw: _,
        } => {
            let (prompt_before_assistant, prompt_after_assistant) =
                get_prompt_according_to_session_status(&session_state);
            let response = call_llm_with_prefix::<M, C>(
                llm_callable,
                prompt_before_assistant,
                prompt_after_assistant,
            )
            .await;
            if is_context_length_exceeded_response(&response) {
                context_length_exceeded_result(question_id, session_state.final_answer.is_some())
            } else {
                let mut actions = Vec::new();
                if session_state.final_answer.is_none()
                    && let Some(boxed_content) = extract_boxed_content(&response)
                {
                    actions.push(TreeAction::AddTrajectoryAction {
                        question_id,
                        action: TrajectoryAction::SubmitFinalAnswer(FinalAnswer::ModelProvided(
                            boxed_content,
                        )),
                    });
                }
                let (step_content_compacted, step_quality) = parse_compactor_response(response);
                actions.push(TreeAction::AddTrajectoryAction {
                    question_id,
                    action: TrajectoryAction::CompactorCompactStep {
                        step_content_compacted,
                        step_quality,
                    },
                });
                actions
            }
        }
        TrajectoryStatus::PlannerUpdatingPlan {
            planner_chosen_mode: _,
            step_content_raw: _,
            step_content_compacted: _,
        } => {
            if session_state.final_answer.is_none() {
                let (prompt_before_assistant, prompt_after_assistant) =
                    get_prompt_according_to_session_status(&session_state);
                let response = call_llm_with_prefix::<M, C>(
                    llm_callable,
                    prompt_before_assistant,
                    prompt_after_assistant,
                )
                .await;
                if is_context_length_exceeded_response(&response) {
                    context_length_exceeded_result(
                        question_id,
                        session_state.final_answer.is_some(),
                    )
                } else {
                    let updated_plan_content = response; // we change to not require the updated plan to be in a markdown code block
                    vec![TreeAction::AddTrajectoryAction {
                        question_id,
                        action: TrajectoryAction::PlannerUpdatePlan(Some(updated_plan_content)),
                    }]
                }
            } else {
                // if final answer already exists, we skip the plan updating and directly add a PlannerUpdatePlan action with None content, which will be treated as a signal to skip updating plan in the reducer
                vec![TreeAction::AddTrajectoryAction {
                    question_id,
                    action: TrajectoryAction::PlannerUpdatePlan(None),
                }]
            }
        }
        TrajectoryStatus::TrajectoryEnded { final_answer } => {
            let mut actions = Vec::new();
            // let display_final_answer = match final_answer {
            //     FinalAnswer::ModelProvided(ans) => ans.clone(),
            //     FinalAnswer::Failure(reason) => format!("Failure: {}", reason),
            // };
            let leaf_node_id = tree
                .current_node_id
                .expect("WorkingOnTrajectory should always have current_node_id when ending");

            let register_leaf_action = TreeAction::RegisterLeaf {
                question_id: tree.question_id,
                node_id: leaf_node_id,
            };

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
            let judge_leaf_correctness_event = TreeAction::JudgeLeafCorrectness {
                question_id: tree.question_id,
                node_id: leaf_node_id,
                correctness_judgment: CorrectnessJudgment {
                    model_answer: final_answer.clone(),
                    correct_answer: tree.reference_answer.clone(),
                    is_correct,
                },
            };
            assert!(
                !tree.leaf_node_ids.contains(&leaf_node_id),
                "The leaf node for the trajectory should not have been registered yet"
            );
            assert!(
                !tree.leaf_node_judgments.contains_key(&leaf_node_id),
                "The leaf node for the trajectory should not have been judged yet"
            );
            actions.push(register_leaf_action);
            actions.push(judge_leaf_correctness_event);
            actions.push(TreeAction::AddTrajectoryAction {
                question_id,
                action: TrajectoryAction::StartDeterminingBranchingNode,
            });
            actions
        }
        TrajectoryStatus::DeterminingBranchingNode => {
            let mut actions = Vec::new();
            let branching_node = determine_branching_node(tree, rng);
            match branching_node {
                Some(branching_node) => actions.push(TreeAction::CreateAndMoveToNode {
                    question_id,
                    parent_id: Some(branching_node),
                }),
                None => {
                    actions.push(TreeAction::TreeComplete { question_id });
                }
            }
            actions
        }
    }
}
