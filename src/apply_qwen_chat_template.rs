use std::sync::LazyLock;

use serde::Serialize;



static QWEN_TEMPLATE_ENVIRONMENT: LazyLock<minijinja::Environment> = LazyLock::new(|| {
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

pub fn apply_qwen_chat_template(user_prompt: &str) -> String {
    let tmpl = QWEN_TEMPLATE_ENVIRONMENT.get_template("chat").unwrap();
    let messages = vec![
        ChatMessage {
            role: "user".into(),
            content: user_prompt.into(),
        },
    ];
    let rendered = tmpl.render(minijinja::context! {
        messages => messages,
        add_generation_prompt => true,
    }).unwrap();
    rendered
}