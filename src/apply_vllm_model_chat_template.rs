use std::sync::LazyLock;

use serde::Serialize;

use crate::deepmath::generate_raw_answers::Model;

static QWEN25_TEMPLATE_ENVIRONMENT: LazyLock<minijinja::Environment> = LazyLock::new(|| {
    let mut env = minijinja::Environment::new();
    let template_src = std::fs::read_to_string("tokenizers/qwen25/chat_template.jinja").unwrap();
    env.add_template_owned("chat", template_src).unwrap();
    env
});

static QWEN3_TEMPLATE_ENVIRONMENT: LazyLock<minijinja::Environment> = LazyLock::new(|| {
    let mut env = minijinja::Environment::new();
    let template_src = std::fs::read_to_string("tokenizers/qwen3/chat_template.jinja").unwrap();
    env.add_template_owned("chat", template_src).unwrap();
    env
});

#[derive(Serialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

pub fn apply_vllm_model_chat_template(
    model: Model,
    user_prompt: &str,
    enable_thinking: bool,
) -> String {
    assert!(
        model.is_qwen(),
        "vLLM chat template only supports Qwen models"
    );

    let template_environment = match model {
        Model::Qwen25_7b => &QWEN25_TEMPLATE_ENVIRONMENT,
        Model::Qwen3_4b | Model::Qwen3_8b | Model::Qwen35_4b => &QWEN3_TEMPLATE_ENVIRONMENT,
        Model::Gpt4o | Model::Gpt5Mini => unreachable!(),
    };
    let tmpl = template_environment.get_template("chat").unwrap();
    let messages = vec![ChatMessage {
        role: "user".into(),
        content: user_prompt.into(),
    }];
    tmpl.render(minijinja::context! {
        messages => messages,
        add_generation_prompt => true,
        enable_thinking => enable_thinking,
    })
    .unwrap()
}
