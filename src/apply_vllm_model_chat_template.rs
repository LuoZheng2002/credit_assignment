use std::sync::LazyLock;

use serde::Serialize;

use crate::deepmath::generate_raw_answers::Model;

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

fn apply_simple_qwen_chatml_template(user_prompt: &str) -> String {
    format!(
        "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
        user_prompt
    )
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

    match model {
        Model::Qwen25_7b => {
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
        Model::Qwen3_4b | Model::Qwen3_8b | Model::Qwen35_4b => {
            assert!(
                !enable_thinking,
                "Qwen3 prefix template path assumes thinking is disabled"
            );
            apply_simple_qwen_chatml_template(user_prompt)
        }
        Model::Gpt4o | Model::Gpt5Mini => unreachable!(),
    }
}
