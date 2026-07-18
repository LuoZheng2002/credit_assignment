use credit_assignment::judge_correctness::{
    DEEPSEEK_CHAT_COMPLETIONS_URL, OPENROUTER_CHAT_COMPLETIONS_URL, USE_OPENROUTER_API,
    extract_boxed_verdict, fetch_judge_evaluation_with_url, judge_answer_task_with_url,
};
use reqwest::Client;

fn judge_api_url() -> &'static str {
    if USE_OPENROUTER_API {
        OPENROUTER_CHAT_COMPLETIONS_URL
    } else {
        DEEPSEEK_CHAT_COMPLETIONS_URL
    }
}

/// Helper: call `fetch_judge_evaluation_with_url` against the real
/// API, panicking on failure (so test failure messages are clear).
async fn call_api(
    client: &Client,
    prompt: &str,
    temperature: f64,
    thinking_enabled: bool,
) -> (String, Option<String>) {
    fetch_judge_evaluation_with_url(
        client,
        prompt,
        judge_api_url(),
        temperature,
        thinking_enabled,
    )
    .await
    .expect("API call should succeed")
}

/// Helper: run `judge_answer_task_with_url` against the real API.
async fn call_judge(
    client: &Client,
    model_answer: &str,
    correct_answer: &str,
    question: &str,
) -> bool {
    judge_answer_task_with_url(
        model_answer.to_string(),
        correct_answer.to_string(),
        question.to_string(),
        client.clone(),
        judge_api_url(),
    )
    .await
}

/// Thinking mode (temperature=0, thinking enabled) should return a valid
/// answer and non-empty reasoning content.
#[tokio::test]
async fn test_thinking_mode() {
    let _ = dotenvy::dotenv();
    let client = Client::new();
    let (content, reasoning) = call_api(
        &client,
        "A student says the answer is '4'. The answer key says '4'. Think step by step and put your verdict in \\boxed{correct} or \\boxed{incorrect}.",
        0.0,
        true,
    )
    .await;
    let lower = content.trim().to_lowercase();
    assert!(
        lower.contains("correct") || lower.contains("incorrect"),
        "thinking mode should return a valid verdict, got: {content:?}"
    );
    assert!(
        reasoning.is_some(),
        "reasoning should be present when thinking is enabled"
    );
}

/// The full judge pipeline should return true when the model answer matches
/// the correct answer.
#[tokio::test]
async fn test_judge_correct_answer() {
    let _ = dotenvy::dotenv();
    let client = Client::new();
    let result = call_judge(&client, "4", "4", "What is 2+2?").await;
    assert!(result, "judge should return true for matching answers");
}

/// The full judge pipeline should return false when the model answer does
/// not match the correct answer.
#[tokio::test]
async fn test_judge_incorrect_answer() {
    let _ = dotenvy::dotenv();
    let client = Client::new();
    let result = call_judge(&client, "5", "4", "What is 2+2?").await;
    assert!(!result, "judge should return false for mismatched answers");
}

/// The full judge pipeline should return true when the model answer matches
/// Realistic end-to-end test using the exact same prompt format as
/// `judge_answer_task_with_url`.  The model answer and reference answer are
/// semantically equivalent but phrased differently ("x equals 5" vs "five"),
/// creating real ambiguity.  Reports the full model output, then verifies
/// the judge returns true.
#[tokio::test]
async fn test_realistic_ambiguous_judgment() {
    let _ = dotenvy::dotenv();
    let client = Client::new();

    let question = "What is the value of x if 3x + 5 = 20?";
    let model_answer = "x equals 5";
    let correct_answer = "five";

    // Exact same prompt construction as judge_answer_task_with_url
    let base_prompt = format!(
        "You are an answer checker that checks a model's answer against the reference answer. \
         Judge if the model's answer is equivalent to the reference answer. \
         Do not attempt to solve the problem yourself, only judge whether the given answer and the reference answer is equivalent. \
         If the model's answer contains units but the reference answer does not, treat them as equivalent if the numerical values are the same. \n\
         The question is: \"{}\". \
         The model's answer is: \"{}\", and the correct answer is: \"{}\".",
        question, model_answer, correct_answer
    );
    let thinking_prompt = format!(
        "{base_prompt} Think step by step about whether the model's answer matches the reference answer. Put your final answer in \\boxed{{correct}} or \\boxed{{incorrect}}."
    );

    // ── Thinking mode (temp=0, thinking enabled) ──
    eprintln!("\n========== THINKING MODE (temp=0) ==========");
    let (content, reasoning) = call_api(&client, &thinking_prompt, 0.0, true).await;
    eprintln!("[content]\n{content}");
    eprintln!("[reasoning] {:?}", reasoning);
    let verdict = extract_boxed_verdict(&content);
    eprintln!("[extracted verdict from \\boxed{{}}] {:?}", verdict);
    assert!(
        verdict.is_some(),
        "thinking mode should produce a \\boxed{{}} response"
    );

    // ── Full judge pipeline ──
    let result = call_judge(&client, model_answer, correct_answer, question).await;
    eprintln!("\n[judge_answer_task_with_url result] {result}");
    assert!(
        result,
        "judge should recognize 'x equals 5' and 'five' as equivalent"
    );
}
