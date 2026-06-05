use clap::{ValueEnum, builder::PossibleValue};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

#[derive(Clone, Debug, Copy)]
pub enum LlmModelName {
    Gpt4o,
    Gemma3_4b,
    Llama31_8b,
    Qwen25_7b,
    Qwen3_06b,
    Qwen3_4b,
    Qwen35_08b,
    Qwen35_4b,
}

impl ValueEnum for LlmModelName {
    fn value_variants<'a>() -> &'a [Self] {
        &[
            Self::Gpt4o,
            Self::Gemma3_4b,
            Self::Llama31_8b,
            Self::Qwen25_7b,
            Self::Qwen3_06b,
            Self::Qwen3_4b,
            Self::Qwen35_08b,
            Self::Qwen35_4b,
        ]
    }

    fn to_possible_value(&self) -> Option<PossibleValue> {
        Some(PossibleValue::new(self.cli_name()))
    }
}

impl Serialize for LlmModelName {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let possible_value = self
            .to_possible_value()
            .expect("LlmModel variant should always have a clap value name");
        serializer.serialize_str(possible_value.get_name())
    }
}

impl<'de> Deserialize<'de> for LlmModelName {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        LlmModelName::from_str(&value, false).map_err(serde::de::Error::custom)
    }
}

impl LlmModelName {
    pub fn cli_name(&self) -> &'static str {
        match self {
            LlmModelName::Gpt4o => "gpt-4o",
            LlmModelName::Gemma3_4b => "gemma-3-4b-it",
            LlmModelName::Llama31_8b => "llama-3.1-8b-instruct",
            LlmModelName::Qwen25_7b => "qwen2.5-7b",
            LlmModelName::Qwen3_06b => "qwen3-0.6b",
            LlmModelName::Qwen3_4b => "qwen3-4b",
            LlmModelName::Qwen35_08b => "qwen3.5-0.8b",
            LlmModelName::Qwen35_4b => "qwen3.5-4b",
        }
    }

    pub fn api_name(&self) -> &'static str {
        match self {
            LlmModelName::Gpt4o => "gpt-4o",
            LlmModelName::Gemma3_4b => "google/gemma-3-4b-it",
            LlmModelName::Llama31_8b => "meta-llama/Llama-3.1-8B-Instruct",
            LlmModelName::Qwen25_7b => "Qwen/Qwen2.5-7B-Instruct",
            LlmModelName::Qwen3_06b => "Qwen/Qwen3-0.6B",
            LlmModelName::Qwen3_4b => "Qwen/Qwen3-4B",
            LlmModelName::Qwen35_08b => "Qwen/Qwen3.5-0.8B",
            LlmModelName::Qwen35_4b => "Qwen/Qwen3.5-4B",
        }
    }

    pub fn is_qwen(&self) -> bool {
        match self {
            LlmModelName::Qwen25_7b
            | LlmModelName::Qwen3_06b
            | LlmModelName::Qwen3_4b
            | LlmModelName::Qwen35_08b
            | LlmModelName::Qwen35_4b => true,
            LlmModelName::Gpt4o
            | LlmModelName::Gemma3_4b
            | LlmModelName::Llama31_8b => false,
        }
    }

    pub fn is_gpt(&self) -> bool {
        match self {
            LlmModelName::Gpt4o => true,
            LlmModelName::Gemma3_4b | LlmModelName::Llama31_8b => false,
            LlmModelName::Qwen25_7b
            | LlmModelName::Qwen3_06b
            | LlmModelName::Qwen3_4b
            | LlmModelName::Qwen35_08b
            | LlmModelName::Qwen35_4b => false,
        }
    }
}
