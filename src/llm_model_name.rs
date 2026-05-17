use clap::{ValueEnum, builder::PossibleValue};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

#[derive(Clone, Debug, Copy)]
pub enum LlmModelName {
    Gpt4o,
    Gpt5Mini,
    Qwen25_7b,
    Qwen3_4b,
    Qwen3_8b,
    Qwen35_4b,
}

impl ValueEnum for LlmModelName {
    fn value_variants<'a>() -> &'a [Self] {
        &[
            Self::Gpt4o,
            Self::Gpt5Mini,
            Self::Qwen25_7b,
            Self::Qwen3_4b,
            Self::Qwen3_8b,
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
            LlmModelName::Gpt5Mini => "gpt-5-mini",
            LlmModelName::Qwen25_7b => "qwen2.5-7b",
            LlmModelName::Qwen3_4b => "qwen3-4b",
            LlmModelName::Qwen3_8b => "qwen3-8b",
            LlmModelName::Qwen35_4b => "qwen3.5-4b",
        }
    }

    pub fn api_name(&self) -> &'static str {
        match self {
            LlmModelName::Gpt4o => "gpt-4o",
            LlmModelName::Gpt5Mini => "gpt-5-mini",
            LlmModelName::Qwen25_7b => "Qwen/Qwen2.5-7B-Instruct",
            LlmModelName::Qwen3_4b => "Qwen/Qwen3-4B",
            LlmModelName::Qwen3_8b => "Qwen/Qwen3-8B",
            LlmModelName::Qwen35_4b => "Qwen/Qwen3.5-4B",
        }
    }

    pub fn is_qwen(&self) -> bool {
        match self {
            LlmModelName::Qwen25_7b
            | LlmModelName::Qwen3_4b
            | LlmModelName::Qwen3_8b
            | LlmModelName::Qwen35_4b => true,
            LlmModelName::Gpt4o | LlmModelName::Gpt5Mini => false,
        }
    }

    pub fn is_gpt(&self) -> bool {
        match self {
            LlmModelName::Gpt4o | LlmModelName::Gpt5Mini => true,
            LlmModelName::Qwen25_7b
            | LlmModelName::Qwen3_4b
            | LlmModelName::Qwen3_8b
            | LlmModelName::Qwen35_4b => false,
        }
    }
}
