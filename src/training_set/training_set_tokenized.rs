use serde::{Deserialize, Serialize};
use tokenizers::Tokenizer;

use crate::{
    direct_answer::generate_raw_answers::LlmModel,
    em::em_schema::short_hyperparameter_hash,
    em::em_types::EmHyperparameters,
    parallel_process_jsonl::{read_json, read_json_lines, write_json, write_jsonl_file},
    training_set::training_set_formatted::{AssetFileTrainingFormatted, TrainingSampleFormatted},
    version_tracking::{AssetFile, Base64Hash, hash_file},
};
// The tokenization should follow the following rules:
// 1. The control tags are <__start_mask__> and <__end_mask_with_eos__>, and tags are removed before tokenization.
// 2. Text inside mask regions has label equal to input_ids, while text outside mask regions has label -100.
// 3. The final formatted content must end with <__end_mask_with_eos__>. For this last end tag,
//    EOS is appended after the full sequence in both input_ids and labels.
// 4. The reconstructed field should follow the same masked format as TrainingSampleFormatted content.

// Here is a concrete example:
// Formatted content:
// User: What is 1+1? Assistant: <__start_mask__>Let me call the tool. <tool_call>What is 1+1?</tool_call> <__end_mask_with_eos__><tool_response>The answer is 2.</tool_response><__start_mask__> Therefore, the answer is 2.<__end_mask_with_eos__>
// Tokenized input_ids (after removing tags, should be in the form of integers in practice):
// ["User:", "What", "is", "1+1?", "Assistant:", "Let", "me", "call", "the", "tool.", "<tool_call>", "What", "is", "1+1?", "</tool_call>", "<tool_response>", "The", "answer", "is", "2.", "</tool_response>", "Therefore,", "the", "answer", "is", "2.", "<END_OF_SEQUENCE>"]
// Labels:
// [null, null, null, null, null, "Let", "me", "call", "the", "tool.", "<tool_call>", "What", "is", "1+1?", "</tool_call>", "<END_OF_SEQUENCE>", null, null, null, null, null, "Therefore,", "the", "answer", "is", "2.", "<END_OF_SEQUENCE>"]

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingSampleTokenized {
    pub question_id: usize,
    pub node_id: usize,
    pub input_ids: Vec<i32>,
    pub labels: Vec<i32>,
    pub reconstructed: String,
    pub input_length: usize,
    pub advantage: f64,
}

pub struct AssetFileTrainingTokenized {
    pub model: LlmModel,
    pub dataset: String,
    pub num_samples: usize,
    pub hyperparameters: EmHyperparameters,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssetFileTrainingTokenizedTracking {
    pub formatted_hash: Base64Hash,
    pub tokenized_schema_version: usize,
}

impl AssetFileTrainingTokenized {
    const START_MASK_TAG: &'static str = "<__start_mask__>";
    const END_MASK_WITH_EOS_TAG: &'static str = "<__end_mask_with_eos__>";
    const END_OF_CONVERSATION_TOKEN: &'static str = "<|im_end|>";
    const IGNORE_LABEL: i32 = -100;
    const TOKENIZED_SCHEMA_VERSION: usize = 6;

    pub fn hyperparameter_hash(&self) -> String {
        short_hyperparameter_hash(&self.hyperparameters)
    }

    pub fn file_path(&self) -> String {
        format!(
            "results/{}/agent/{}_training_tokenized_{}_{}.jsonl",
            self.model.cli_name(),
            self.dataset,
            self.num_samples,
            self.hyperparameter_hash(),
        )
    }

    pub fn version_tracking_path(&self) -> String {
        format!(
            "results_version_tracking/{}/agent/{}_training_tokenized_{}_{}.version.json",
            self.model.cli_name(),
            self.dataset,
            self.num_samples,
            self.hyperparameter_hash(),
        )
    }

    fn load_tokenizer(&self) -> Tokenizer {
        // assert!(
        //     self.model.is_qwen(),
        //     "Training tokenization currently supports Qwen models only"
        // );
        let model = if self.model.is_qwen() {
            self.model
        } else {
            println!("Warning: Training tokenization currently supports Qwen models only, but received {}", self.model.cli_name());
            LlmModel::Qwen25_7b
        };
        Tokenizer::from_pretrained(model.api_name(), None).unwrap()
    }

    fn formatted_to_tokenized_with_tokenizer(
        &self,
        formatted_sample: &TrainingSampleFormatted,
        tokenizer: &Tokenizer,
    ) -> TrainingSampleTokenized {
        #[derive(Clone, Copy, PartialEq, Eq)]
        enum SegmentMode {
            Unmasked,
            Masked,
        }

        struct Segment {
            mode: SegmentMode,
            text: String,
        }

        let end_of_conversation_id_u32 = tokenizer
            .token_to_id(Self::END_OF_CONVERSATION_TOKEN)
            .unwrap();
        assert!(
            end_of_conversation_id_u32 <= i32::MAX as u32,
            "End-of-conversation token id must fit in i32"
        );
        let end_of_conversation_id = end_of_conversation_id_u32 as i32;

        let original = formatted_sample.content_formatted.as_str();
        let mut cursor = 0usize;
        let mut in_mask = false;
        let mut segments: Vec<Segment> = Vec::new();

        let tags = [Self::START_MASK_TAG, Self::END_MASK_WITH_EOS_TAG];

        while cursor < original.len() {
            let mut next_tag: Option<(usize, &'static str)> = None;
            for tag in tags {
                if let Some(relative_pos) = original[cursor..].find(tag) {
                    let absolute_pos = cursor + relative_pos;
                    match next_tag {
                        Some((existing_pos, _)) if absolute_pos >= existing_pos => {}
                        _ => next_tag = Some((absolute_pos, tag)),
                    }
                }
            }

            let next_text_end = match next_tag {
                Some((pos, _)) => pos,
                None => original.len(),
            };
            if next_text_end > cursor {
                let text_chunk = &original[cursor..next_text_end];
                let mode = if in_mask {
                    SegmentMode::Masked
                } else {
                    SegmentMode::Unmasked
                };
                segments.push(Segment {
                    mode,
                    text: text_chunk.to_string(),
                });
            }

            let Some((tag_pos, tag)) = next_tag else {
                break;
            };
            assert_eq!(tag_pos, next_text_end, "Tag parsing cursor mismatch");

            if tag == Self::START_MASK_TAG {
                assert!(
                    !in_mask,
                    "Unexpected nested {} in formatted sample (question_id={}, node_id={})",
                    Self::START_MASK_TAG,
                    formatted_sample.question_id,
                    formatted_sample.node_id
                );
                in_mask = true;
            } else {
                assert_eq!(tag, Self::END_MASK_WITH_EOS_TAG, "Unexpected tag in parser");
                assert!(
                    in_mask,
                    "Missing {} before {} in formatted sample (question_id={}, node_id={})",
                    Self::START_MASK_TAG,
                    Self::END_MASK_WITH_EOS_TAG,
                    formatted_sample.question_id,
                    formatted_sample.node_id
                );
                in_mask = false;
            }

            cursor = tag_pos + tag.len();
        }

        assert!(
            !in_mask,
            "Formatted sample must close {} (question_id={}, node_id={})",
            Self::END_MASK_WITH_EOS_TAG,
            formatted_sample.question_id,
            formatted_sample.node_id
        );
        assert!(
            original.ends_with(Self::END_MASK_WITH_EOS_TAG),
            "Formatted sample must end with {} (question_id={}, node_id={})",
            Self::END_MASK_WITH_EOS_TAG,
            formatted_sample.question_id,
            formatted_sample.node_id
        );
        assert!(
            original.contains(Self::START_MASK_TAG),
            "Formatted sample must contain at least one mask segment"
        );
        assert!(
            !segments.is_empty(),
            "Formatted sample cannot produce empty segment list"
        );

        let mut input_ids: Vec<i32> = Vec::new();
        let mut labels: Vec<i32> = Vec::new();
        let mut masked_segment_end_indices: Vec<usize> = Vec::new();

        for segment in &segments {
            if segment.text.is_empty() {
                continue;
            }
            let segment_encoding = tokenizer.encode(segment.text.clone(), false).unwrap();
            let segment_ids: Vec<i32> = segment_encoding
                .get_ids()
                .iter()
                .map(|id| {
                    assert!(*id <= i32::MAX as u32, "Token id must fit in i32");
                    *id as i32
                })
                .collect();
            if segment_ids.is_empty() {
                continue;
            }
            for token_id in &segment_ids {
                input_ids.push(*token_id);
                labels.push(match segment.mode {
                    SegmentMode::Masked => *token_id,
                    SegmentMode::Unmasked => Self::IGNORE_LABEL,
                });
            }
            if segment.mode == SegmentMode::Masked {
                masked_segment_end_indices.push(input_ids.len() - 1);
            }
        }

        assert!(!input_ids.is_empty(), "Tokenized input must be non-empty");
        assert_eq!(
            labels.len(),
            input_ids.len(),
            "labels must align with input_ids"
        );
        assert!(
            !masked_segment_end_indices.is_empty(),
            "At least one masked segment must contain tokenized text"
        );

        for token_index in masked_segment_end_indices
            .iter()
            .take(masked_segment_end_indices.len() - 1)
        {
            labels[*token_index] = end_of_conversation_id;
        }

        input_ids.push(end_of_conversation_id);
        labels.push(end_of_conversation_id);

        let mut reconstructed = String::new();
        for segment in &segments {
            match segment.mode {
                SegmentMode::Unmasked => reconstructed.push_str(&segment.text),
                SegmentMode::Masked => {
                    reconstructed.push_str(Self::START_MASK_TAG);
                    reconstructed.push_str(&segment.text);
                    reconstructed.push_str(Self::END_MASK_WITH_EOS_TAG);
                }
            }
        }

        assert_eq!(
            reconstructed, formatted_sample.content_formatted,
            "Reconstructed content must match formatted content"
        );

        let input_length = input_ids.len();

        TrainingSampleTokenized {
            question_id: formatted_sample.question_id,
            node_id: formatted_sample.node_id,
            input_ids,
            labels,
            reconstructed,
            input_length,
            advantage: formatted_sample.advantage,
        }
    }

    pub fn formatted_to_tokenized(
        &self,
        formatted_sample: &TrainingSampleFormatted,
    ) -> TrainingSampleTokenized {
        let tokenizer = self.load_tokenizer();
        self.formatted_to_tokenized_with_tokenizer(formatted_sample, &tokenizer)
    }

    pub fn generate_tokenized_samples(
        &self,
        formatted_samples: &[TrainingSampleFormatted],
    ) -> Vec<TrainingSampleTokenized> {
        let tokenizer = self.load_tokenizer();
        formatted_samples
            .iter()
            .map(|sample| self.formatted_to_tokenized_with_tokenizer(sample, &tokenizer))
            .collect()
    }
}

// use the same style as AssetFileAdvantageComposition
impl AssetFile for AssetFileTrainingTokenized {
    type FileModel = Vec<TrainingSampleTokenized>;

    fn fetch(&self) -> Self::FileModel {
        self.synchronize();
        read_json_lines(self.file_path()).unwrap()
    }

    fn synchronize(&self) -> crate::version_tracking::Base64Hash {
        let asset_file_training_formatted = AssetFileTrainingFormatted {
            model: self.model,
            dataset: self.dataset.clone(),
            num_samples: self.num_samples,
            hyperparameters: self.hyperparameters.clone(),
        };
        let formatted_hash = asset_file_training_formatted.synchronize();

        let tracking_content =
            match read_json::<AssetFileTrainingTokenizedTracking>(self.version_tracking_path()) {
                Ok(mut tracking) => {
                    if tracking.formatted_hash != formatted_hash
                        || tracking.tokenized_schema_version != Self::TOKENIZED_SCHEMA_VERSION
                    {
                        let formatted_samples = asset_file_training_formatted.fetch();
                        let tokenized_samples = self.generate_tokenized_samples(&formatted_samples);
                        write_jsonl_file(self.file_path(), &tokenized_samples).unwrap();
                        tracking.formatted_hash = formatted_hash.clone();
                        tracking.tokenized_schema_version = Self::TOKENIZED_SCHEMA_VERSION;
                    }
                    tracking
                }
                Err(_) => {
                    let formatted_samples = asset_file_training_formatted.fetch();
                    let tokenized_samples = self.generate_tokenized_samples(&formatted_samples);
                    write_jsonl_file(self.file_path(), &tokenized_samples).unwrap();
                    AssetFileTrainingTokenizedTracking {
                        formatted_hash,
                        tokenized_schema_version: Self::TOKENIZED_SCHEMA_VERSION,
                    }
                }
            };
        write_json(self.version_tracking_path(), &tracking_content).unwrap();
        hash_file(self.file_path()).unwrap()
    }
}
