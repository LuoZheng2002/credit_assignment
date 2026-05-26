use std::collections::BTreeMap;

use clap::ValueEnum;
use research_utility::{
    asset_file::{AssetFile, Base64Hash, hash_file},
    sqlite_store::SqliteStore,
};
use serde::{Deserialize, Serialize};

use crate::{
    direct_tool::{
        direct_rollout_config::DirectRolloutConfig,
        direct_tree::{DirectTree, Segment, SegmentContent, SegmentId},
        direct_tree_action_log::{AssetFileDirectTreeActionLogs, DirectTreeActionLog},
        hybrid_dataset::HybridDatasetQuestion,
        posterior_calculation_config::PosteriorCalculationConfig,
    },
    json_line_util::{read_json, write_json},
    llm_model::{LlmModelMarker, LlmModelName},
};

pub async fn rollout_logs_to_training_trajectories<M: LlmModelMarker>(
    action_log_store: SqliteStore<usize, DirectTreeActionLog>,
) -> Vec<DirectTrainingTrajectory<M>> {
    // iterate through all action logs
    let mut keys = action_log_store.get_keys().await.unwrap();
    keys.sort(); // ensure deterministic order
    for key in keys {
        let action_log = action_log_store.get(key).await.unwrap().unwrap();
        // convert each action log into a training trajectory
    }

    todo!()
}

fn action_log_to_candidate_trajectories<M: LlmModelMarker>(
    action_log: DirectTreeActionLog,
) -> Vec<DirectTrainingTrajectory<M>> {
    let tree = DirectTree::<M>::from_action_log(&action_log);
    let segment_posteriors = tree.calculate_segment_posteriors(None);
    let segment_advantages_unnormalized = segment_posteriors
        .into_iter()
        .map(|(segment_id, posterior)| {
            (segment_id, posterior.mean / posterior.log_std.exp()) // we use the mean/std as the unnormalized advantage
        })
        .collect::<Vec<(SegmentId, f32)>>();
    let segment_advantages_mean = segment_advantages_unnormalized
        .iter()
        .map(|(_, advantage)| *advantage)
        .sum::<f32>()
        / segment_advantages_unnormalized.len() as f32;
    let segment_advantages_std = (segment_advantages_unnormalized
        .iter()
        .map(|(_, advantage)| {
            let diff = *advantage - segment_advantages_mean;
            diff * diff
        })
        .sum::<f32>()
        / segment_advantages_unnormalized.len() as f32)
        .sqrt();
    let mut segment_advantages = segment_advantages_unnormalized
        .into_iter()
        .map(|(segment_id, advantage)| {
            let normalized_advantage = if segment_advantages_std > 0.0 {
                let mut shifted_advantage = advantage - segment_advantages_mean;
                if shifted_advantage * advantage < 0.0 {
                    shifted_advantage = 0.0; // if the advantage is on the opposite side of the mean compared to the unnormalized advantage, we set it to 0 to avoid hurting the training
                }
                shifted_advantage / segment_advantages_std
            } else {
                0.0
            };
            (segment_id, normalized_advantage)
        })
        .collect::<BTreeMap<SegmentId, f32>>();
    // to do: we need to first find the trajectories with the most average advantage
    // after taking it, we need to set advantages for taken segments to be 0.0
    // then we find the trajectory with the most average advantage among the remaining ones
    
    let mut trajectories: Vec<DirectTrainingTrajectory<M>> = Vec::new();
    // for each leaf segment, we create a trajectory and calculate the advantage
    for leaf_segment_id in tree.trunk_leaf_segments.iter() {
        let mut current_segment_id: Option<SegmentId> = Some(*leaf_segment_id);
        let mut segments: Vec<Segment> = Vec::new();
        while let Some(segment_id) = current_segment_id {
            let segment = tree.segments.get(&segment_id).unwrap();
            segments.push(segment.clone());
            current_segment_id = segment.parent_id;
        }
        segments.reverse(); // we want the segments to be in the order from root to leaf
        let mut advantage_sum = 0.0;
        let mut input_ids: Vec<i32> = Vec::new();
        let mut labels: Vec<i32> = Vec::new();
        let mut advantages: Vec<f32> = Vec::new();
        for segment in segments.iter() {
            let segment_advantage = segment_advantages.get(&segment.segment_id).unwrap();
            advantage_sum += segment_advantage;
            for content in segment.content.iter() {
                match content {
                    SegmentContent::Prompt(token_array) | SegmentContent::ToolResponse(token_array) => {
                        input_ids.extend(token_array.tokens.iter());
                        labels.extend(vec![-100; token_array.tokens.len()]); // we set the labels for the prompt tokens to -100 so that they will be ignored in the loss calculation
                        advantages.extend(vec![*segment_advantage; token_array.tokens.len()]); // we assign the same advantage to all tokens in the segment
                    },
                    SegmentContent::ReasoningOrToolCall { tokens, complete } => {
                        input_ids.extend(tokens.tokens.iter());
                        labels.extend(tokens.tokens.iter());
                        advantages.extend(vec![*segment_advantage; tokens.tokens.len()]);
                    }
                }
            }
        }
        let average_segment_advantage = advantage_sum / segments.len() as f32;
        trajectories.push(DirectTrainingTrajectory {
            question: tree.question.clone(),
            input_ids,
            labels,
            advantages,
            average_segment_advantage,
            _phantom: std::marker::PhantomData::<M>,
        });
    }
    trajectories
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DirectTrainingTrajectory<M: LlmModelMarker> {
    pub question: HybridDatasetQuestion,
    pub input_ids: Vec<i32>,
    pub labels: Vec<i32>, // we may not need to let model learn to stop after tool_wait or end since our framework already handled this
    pub advantages: Vec<f32>,
    pub average_segment_advantage: f32,
    #[serde(skip)]
    pub _phantom: std::marker::PhantomData<M>,
}

pub struct AssetFileTrainingTrajectories<M: LlmModelMarker> {
    // pub model: LlmModelName,
    pub config_nickname: String,
    pub rollout_config: DirectRolloutConfig,
    pub posterior_calculation_config: PosteriorCalculationConfig,
    pub _phantom: std::marker::PhantomData<M>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct AssetFileTrainingTrajectoriesTracking {
    pub rollout_log_hash: Base64Hash,
    pub config_nickname: String,
    pub rollout_config: DirectRolloutConfig,
    pub posterior_calculation_config: PosteriorCalculationConfig,
}

impl<M: LlmModelMarker> AssetFileTrainingTrajectories<M> {
    fn to_short_hash(&self) -> String {
        let serialized = serde_json::to_vec(&(
            &&self.config_nickname,
            &self.rollout_config,
            &self.posterior_calculation_config,
        ))
        .unwrap();
        let hash = blake3::hash(&serialized);
        let short_hash = hex::encode(&hash.as_bytes()[..4]); // Take the first 4 bytes for a shorter hash
        short_hash
    }
    fn model_name(&self) -> LlmModelName {
        LlmModelName::from_str(M::CLI_NAME, true).unwrap()
    }
    pub fn file_path(&self) -> String {
        format!(
            "results/{}/training_trajectories_{}_{}.sqlite",
            M::CLI_NAME,
            self.config_nickname,
            self.to_short_hash()
        )
    }
    fn version_tracking_path(&self) -> String {
        format!(
            "results_version_tracking/{}/training_trajectories_{}_{}_tracking.json",
            M::CLI_NAME,
            self.config_nickname,
            self.to_short_hash()
        )
    }
}

#[async_trait::async_trait]
impl<M: LlmModelMarker> AssetFile for AssetFileTrainingTrajectories<M> {
    type FileModel = SqliteStore<usize, DirectTrainingTrajectory<M>>;
    async fn fetch(&self) -> Self::FileModel {
        self.synchronize().await;
        SqliteStore::<usize, DirectTrainingTrajectory<M>>::assume_initialized(self.file_path())
            .await
    }
    async fn synchronize(&self) -> Base64Hash {
        let asset_file_rollout_logs = AssetFileDirectTreeActionLogs {
            model: self.model_name(),
            nickname: self.config_nickname.clone(),
            rollout_config: self.rollout_config.clone(),
            posterior_calculation_config: self.posterior_calculation_config.clone(),
        };
        let rollout_log_hash = asset_file_rollout_logs.synchronize().await;
        let new_tracking_content = AssetFileTrainingTrajectoriesTracking {
            rollout_log_hash,
            config_nickname: self.config_nickname.clone(),
            rollout_config: self.rollout_config.clone(),
            posterior_calculation_config: self.posterior_calculation_config.clone(),
        };
        let stale = if let Ok(tracking_content) =
            read_json::<AssetFileTrainingTrajectoriesTracking>(self.version_tracking_path())
        {
            tracking_content != new_tracking_content
        } else {
            true
        };
        if stale {
            // if so, we first delete the target file
            if std::path::Path::new(&self.file_path()).exists() {
                std::fs::remove_file(&self.file_path()).unwrap();
            }
            // initialize database
            let db =
                SqliteStore::<usize, DirectTrainingTrajectory<M>>::initialize(self.file_path())
                    .await;
            let rollout_logs = asset_file_rollout_logs.fetch().await;
            let training_trajectories =
                rollout_logs_to_training_trajectories::<M>(rollout_logs).await;
            for (i, trajectory) in training_trajectories.into_iter().enumerate() {
                db.upsert(i, &trajectory).await.unwrap();
            }
        }
        write_json(self.version_tracking_path(), &new_tracking_content).unwrap();
        hash_file(self.file_path()).expect("Failed to hash training trajectories file")
    }
}
