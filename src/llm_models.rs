use serde::{Deserialize, Serialize};
pub trait LlmModelMarker {
    type StringOrTokenArray: Serialize + for<'de> Deserialize<'de> + Clone + std::fmt::Debug;
}
pub struct Qwe25;
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Qwen25TokenArray {
    pub tokens: Vec<i32>,
    pub decoded_string: String,
}
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Qwen3TokenArray {
    pub tokens: Vec<i32>,
    pub decoded_string: String,
}
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Qwen35TokenArray {
    pub tokens: Vec<i32>,
    pub decoded_string: String,
}
impl LlmModelMarker for Qwe25 {
    type StringOrTokenArray = Qwen25TokenArray;
}
pub struct Qwen3;
impl LlmModelMarker for Qwen3 {
    type StringOrTokenArray = Qwen3TokenArray;
}
pub struct Qwen35;
impl LlmModelMarker for Qwen35 {
    type StringOrTokenArray = Qwen35TokenArray;
}
pub struct Gpt4o;
impl LlmModelMarker for Gpt4o {
    type StringOrTokenArray = String;
}