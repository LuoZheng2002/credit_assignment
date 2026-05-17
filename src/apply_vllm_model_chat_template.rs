use std::sync::LazyLock;

use serde::Serialize;

use crate::llm_model_name::LlmModelName;

static QWEN25_TEMPLATE_ENVIRONMENT: LazyLock<minijinja::Environment> = LazyLock::new(|| {
    let mut env = minijinja::Environment::new();
    let template_src = std::fs::read_to_string("tokenizers/qwen25/chat_template.jinja").unwrap();
    env.add_template_owned("chat", template_src).unwrap();
    env
});

#[derive(Serialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

fn apply_simple_qwen_chatml_template(user_prompt: &str, enable_thinking: bool) -> String {
    if enable_thinking {
        format!(
            "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
            user_prompt
        )
    } else {
        format!(
            "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
            user_prompt
        )
    }
}

pub fn apply_vllm_model_chat_template(
    model: LlmModelName,
    user_prompt: &str,
    enable_thinking: bool,
) -> String {
    // assert!(
    //     model.is_qwen(),
    //     "vLLM chat template only supports Qwen models"
    // );
    let model = if model.is_qwen() {
        model
    } else {
        println!(
            "Warning: vLLM chat template only supports Qwen models, but received {}",
            model.cli_name()
        );
        LlmModelName::Qwen25_7b
    };

    match model {
        LlmModelName::Qwen25_7b => {
            let tmpl = QWEN25_TEMPLATE_ENVIRONMENT.get_template("chat").unwrap();
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
        // NOTE: Official Qwen3 templates use Python-style string methods (e.g. startswith)
        // that MiniJinja does not support. Use a minimal ChatML template for prefix mode.
        LlmModelName::Qwen3_4b | LlmModelName::Qwen3_8b | LlmModelName::Qwen35_4b => {
            apply_simple_qwen_chatml_template(user_prompt, enable_thinking)
        }
        LlmModelName::Gpt4o | LlmModelName::Gpt5Mini => unreachable!(),
    }
}
