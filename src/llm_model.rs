use clap::ValueEnum;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

#[derive(clap::ValueEnum, Clone, Debug, Copy)]
pub enum LlmModel {
    #[value(name = "gpt-4o")]
    Gpt4o,
    #[value(name = "gpt-5-mini")]
    Gpt5Mini,
    #[value(name = "qwen2.5-7b")]
    Qwen25_7b,
    #[value(name = "qwen3-4b")]
    Qwen3_4b,
    #[value(name = "qwen3-8b")]
    Qwen3_8b,
    #[value(name = "qwen3.5-4b")]
    Qwen35_4b,
}

impl Serialize for LlmModel {
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

impl<'de> Deserialize<'de> for LlmModel {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        LlmModel::from_str(&value, false).map_err(serde::de::Error::custom)
    }
}

impl LlmModel {
    pub fn cli_name(&self) -> &'static str {
        match self {
            LlmModel::Gpt4o => "gpt-4o",
            LlmModel::Gpt5Mini => "gpt-5-mini",
            LlmModel::Qwen25_7b => "qwen2.5-7b",
            LlmModel::Qwen3_4b => "qwen3-4b",
            LlmModel::Qwen3_8b => "qwen3-8b",
            LlmModel::Qwen35_4b => "qwen3.5-4b",
        }
    }

    pub fn api_name(&self) -> &'static str {
        match self {
            LlmModel::Gpt4o => "gpt-4o",
            LlmModel::Gpt5Mini => "gpt-5-mini",
            LlmModel::Qwen25_7b => "Qwen/Qwen2.5-7B-Instruct",
            LlmModel::Qwen3_4b => "Qwen/Qwen3-4B",
            LlmModel::Qwen3_8b => "Qwen/Qwen3-8B",
            LlmModel::Qwen35_4b => "Qwen/Qwen3.5-4B",
        }
    }

    pub fn is_qwen(&self) -> bool {
        match self {
            LlmModel::Qwen25_7b | LlmModel::Qwen3_4b | LlmModel::Qwen3_8b | LlmModel::Qwen35_4b => {
                true
            }
            LlmModel::Gpt4o | LlmModel::Gpt5Mini => false,
        }
    }

    pub fn is_gpt(&self) -> bool {
        match self {
            LlmModel::Gpt4o | LlmModel::Gpt5Mini => true,
            LlmModel::Qwen25_7b | LlmModel::Qwen3_4b | LlmModel::Qwen3_8b | LlmModel::Qwen35_4b => {
                false
            }
        }
    }
}
