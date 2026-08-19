use std::{
    collections::BTreeMap,
    fs::{self, File, OpenOptions},
    io::{BufRead, BufReader, Write},
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use futures::{StreamExt, stream};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use tokio::{sync::Semaphore, task::JoinHandle, time::sleep};

use crate::judge_correctness::{extract_boxed_verdict, fetch_judge_evaluation_for_model};

pub const DEFAULT_CACHE_CHUNK_QUESTION_COUNT: usize = 1000;
pub const DEFAULT_REQUEST_CONCURRENCY_PER_MODEL: usize = 200;
pub const DEFAULT_CACHE_VERSION: &str = "judgment-cache-v1";

const PHASE_1_MODELS: [&str; 3] = [
    "deepseek/deepseek-v4-flash",
    "qwen/qwen3-32b",
    "google/gemini-2.5-flash-lite",
];
const PHASE_2_MODELS: [&str; 2] = ["deepseek/deepseek-v4-pro", "openai/gpt-4.1-mini"];
const PHASE_3_MODEL: &str = "openai/gpt-5-mini";
const JUDGE_ATTEMPTS: usize = 3;
const CACHE_FLUSH_INTERVAL: usize = 20;
const CACHE_LOCK_WAIT_SECS: u64 = 5;
const CACHE_LOCK_HEARTBEAT_SECS: u64 = 30;
const CACHE_LOCK_STALE_SECS: u64 = 30 * 60;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub struct JudgmentCacheKey {
    pub cache_version: String,
    pub split: String,
    pub flat_id: usize,
    pub model_answer: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JudgingRequestRecord {
    #[serde(default)]
    pub artifact_id: Option<String>,
    pub split: String,
    pub flat_id: usize,
    pub question: String,
    pub correct_answer: String,
    pub model_answer: String,
    #[serde(default)]
    pub metadata: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JudgeModelOutput {
    pub model: String,
    pub verdict: bool,
    pub raw_output: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JudgmentCacheRecord {
    pub key: JudgmentCacheKey,
    pub correct_answer: String,
    pub is_correct: bool,
    pub decision_phase: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub judge_outputs: Vec<JudgeModelOutput>,
    pub updated_unix_secs: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JudgingOutputRecord {
    pub request: JudgingRequestRecord,
    pub is_correct: bool,
    pub decision_phase: String,
    pub cache_hit: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct JudgeModelThroughputStats {
    pub num_calls: usize,
    pub total_model_elapsed_secs: f64,
    pub mean_model_latency_secs: f64,
    pub throughput_per_model_elapsed_sec: f64,
    pub throughput_per_job_elapsed_sec: f64,
}

#[derive(Debug, Clone)]
struct JudgeModelTiming {
    model: String,
    elapsed_secs: f64,
}

#[derive(Debug, Clone)]
struct UncachedJudgmentResult {
    record: JudgmentCacheRecord,
    model_timings: Vec<JudgeModelTiming>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct JudgingSummary {
    pub total_requests: usize,
    pub cache_hits: usize,
    pub newly_judged: usize,
    pub cache_hit_rate: f64,
    pub newly_judged_rate: f64,
    pub cache_chunks_processed: usize,
    pub job_elapsed_secs: f64,
    pub judge_model_throughput: BTreeMap<String, JudgeModelThroughputStats>,
    pub phase1_unanimous: usize,
    pub phase2_agreement: usize,
    pub phase3_escalations: usize,
    pub exact_matches: usize,
    pub failures_marked_incorrect: usize,
}

impl JudgingSummary {
    fn record_model_timing(&mut self, timing: JudgeModelTiming) {
        let stats = self.judge_model_throughput.entry(timing.model).or_default();
        stats.num_calls += 1;
        stats.total_model_elapsed_secs += timing.elapsed_secs;
    }

    fn finalize_rates(&mut self, job_elapsed_secs: f64) {
        self.job_elapsed_secs = job_elapsed_secs;
        if self.total_requests == 0 {
            self.cache_hit_rate = 0.0;
            self.newly_judged_rate = 0.0;
        } else {
            self.cache_hit_rate = self.cache_hits as f64 / self.total_requests as f64;
            self.newly_judged_rate = self.newly_judged as f64 / self.total_requests as f64;
        }
        for stats in self.judge_model_throughput.values_mut() {
            if stats.num_calls == 0 {
                continue;
            }
            stats.mean_model_latency_secs = stats.total_model_elapsed_secs / stats.num_calls as f64;
            if stats.total_model_elapsed_secs > 0.0 {
                stats.throughput_per_model_elapsed_sec =
                    stats.num_calls as f64 / stats.total_model_elapsed_secs;
            }
            if job_elapsed_secs > 0.0 {
                stats.throughput_per_job_elapsed_sec = stats.num_calls as f64 / job_elapsed_secs;
            }
        }
    }
}

pub fn judgment_cache_key(
    cache_version: &str,
    split: &str,
    flat_id: usize,
    model_answer: &str,
) -> JudgmentCacheKey {
    JudgmentCacheKey {
        cache_version: cache_version.to_string(),
        split: split.to_string(),
        flat_id,
        model_answer: model_answer.to_string(),
    }
}

pub fn cache_chunk_index(flat_id: usize, chunk_question_count: usize) -> usize {
    flat_id / chunk_question_count
}

pub fn cache_chunk_path(
    cache_dir: impl AsRef<Path>,
    split: &str,
    flat_id: usize,
    chunk_question_count: usize,
) -> PathBuf {
    cache_dir.as_ref().join(split).join(format!(
        "judgment_cache_chunk_{:08}.jsonl",
        cache_chunk_index(flat_id, chunk_question_count)
    ))
}

fn now_unix_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn build_judge_prompt(question: &str, model_answer: &str, correct_answer: &str) -> String {
    format!(
        "You are an answer checker that checks a model's answer against the reference answer. Judge if the model's answer is equivalent to the reference answer. \
Do not attempt to solve the problem yourself, only judge whether the given answer and the reference answer is equivalent. \
If the model's answer contains units but the reference answer does not, treat them as equivalent if the numerical values are the same. \n\
Your first line must be exactly one of these two strings: \\boxed{{correct}} or \\boxed{{incorrect}}. \
After that first line, you may add a brief explanation. \n\
The question is: \"{}\". \
The model's answer is: \"{}\", and the correct answer is: \"{}\". \
Do not solve the original problem from scratch. Compare only the model answer against the reference answer.",
        question, model_answer, correct_answer
    )
}

async fn judge_with_model(
    client: &Client,
    model: &str,
    question: &str,
    model_answer: &str,
    correct_answer: &str,
) -> Result<JudgeModelOutput, String> {
    let mut prompt = build_judge_prompt(question, model_answer, correct_answer);
    if model.contains("qwen") {
        prompt.push_str("\n/no_think");
    }
    let mut last_error = None;
    for attempt in 0..JUDGE_ATTEMPTS {
        let temperature = if attempt == 0 { 0.0 } else { 0.7 };
        let thinking_enabled = model == PHASE_3_MODEL;
        match fetch_judge_evaluation_for_model(
            client,
            model,
            &prompt,
            temperature,
            thinking_enabled,
        )
        .await
        {
            Ok((raw_output, reasoning)) => {
                if let Some(verdict) = extract_boxed_verdict(&raw_output) {
                    let verdict_lower = verdict.to_lowercase();
                    if verdict_lower.contains("incorrect") {
                        return Ok(JudgeModelOutput {
                            model: model.to_string(),
                            verdict: false,
                            raw_output,
                            reasoning,
                        });
                    }
                    if verdict_lower.contains("correct") {
                        return Ok(JudgeModelOutput {
                            model: model.to_string(),
                            verdict: true,
                            raw_output,
                            reasoning,
                        });
                    }
                    last_error = Some(format!(
                        "boxed verdict was neither correct nor incorrect: {verdict}"
                    ));
                } else {
                    last_error = Some(format!("missing boxed verdict in response: {raw_output}"));
                }
            }
            Err(error) => {
                if is_fatal_judging_error(&error) {
                    return Err(format!(
                        "fatal judging API error from model {model}; aborting without caching a verdict: {error}"
                    ));
                }
                last_error = Some(error);
            }
        }
        if attempt + 1 < JUDGE_ATTEMPTS {
            sleep(Duration::from_secs(1)).await;
        }
    }
    let error = last_error.unwrap_or_else(|| "unknown error".to_string());
    Err(format!(
        "judge model {model} failed after {JUDGE_ATTEMPTS} attempts; aborting without caching a verdict: {error}"
    ))
}

fn is_fatal_judging_error(error: &str) -> bool {
    let lower = error.to_ascii_lowercase();
    lower.contains("insufficient credit")
        || lower.contains("insufficient credits")
        || lower.contains("insufficient balance")
        || lower.contains("not enough credit")
        || lower.contains("not enough credits")
        || lower.contains("not enough balance")
        || lower.contains("402")
        || lower.contains("payment required")
        || lower.contains("quota")
        || lower.contains("rate limit")
        || lower.contains("rate_limit")
}

async fn judge_with_model_timed(
    client: &Client,
    model: &str,
    question: &str,
    model_answer: &str,
    correct_answer: &str,
) -> Result<(JudgeModelOutput, JudgeModelTiming), String> {
    let start = Instant::now();
    let output = judge_with_model(client, model, question, model_answer, correct_answer).await?;
    let timing = JudgeModelTiming {
        model: model.to_string(),
        elapsed_secs: start.elapsed().as_secs_f64(),
    };
    Ok((output, timing))
}

async fn judge_with_models_timed(
    client: &Client,
    models: &[&str],
    question: &str,
    model_answer: &str,
    correct_answer: &str,
) -> Result<(Vec<JudgeModelOutput>, Vec<JudgeModelTiming>), String> {
    let results = futures::future::try_join_all(models.iter().map(|model| {
        judge_with_model_timed(client, model, question, model_answer, correct_answer)
    }))
    .await?;
    let mut outputs = Vec::with_capacity(results.len());
    let mut timings = Vec::with_capacity(results.len());
    for (output, timing) in results {
        outputs.push(output);
        timings.push(timing);
    }
    Ok((outputs, timings))
}

async fn judge_uncached_request(
    client: &Client,
    request: &JudgingRequestRecord,
) -> Result<UncachedJudgmentResult, String> {
    if !request.model_answer.trim().is_empty()
        && request.model_answer.trim() == request.correct_answer.trim()
    {
        return Ok(UncachedJudgmentResult {
            record: JudgmentCacheRecord {
                key: judgment_cache_key("", &request.split, request.flat_id, &request.model_answer),
                correct_answer: request.correct_answer.clone(),
                is_correct: true,
                decision_phase: "exact_match".to_string(),
                judge_outputs: Vec::new(),
                updated_unix_secs: now_unix_secs(),
            },
            model_timings: Vec::new(),
        });
    }

    if request.model_answer.trim().is_empty() {
        return Ok(UncachedJudgmentResult {
            record: JudgmentCacheRecord {
                key: judgment_cache_key("", &request.split, request.flat_id, &request.model_answer),
                correct_answer: request.correct_answer.clone(),
                is_correct: false,
                decision_phase: "empty_answer".to_string(),
                judge_outputs: Vec::new(),
                updated_unix_secs: now_unix_secs(),
            },
            model_timings: Vec::new(),
        });
    }

    let mut outputs = Vec::new();
    let mut model_timings = Vec::new();
    let (phase1_outputs, phase1_timings) = judge_with_models_timed(
        client,
        &PHASE_1_MODELS,
        &request.question,
        &request.model_answer,
        &request.correct_answer,
    )
    .await?;
    outputs.extend(phase1_outputs);
    model_timings.extend(phase1_timings);
    if outputs
        .iter()
        .all(|output| output.verdict == outputs[0].verdict)
    {
        return Ok(UncachedJudgmentResult {
            record: JudgmentCacheRecord {
                key: judgment_cache_key("", &request.split, request.flat_id, &request.model_answer),
                correct_answer: request.correct_answer.clone(),
                is_correct: outputs[0].verdict,
                decision_phase: "phase1_unanimous".to_string(),
                judge_outputs: Vec::new(),
                updated_unix_secs: now_unix_secs(),
            },
            model_timings,
        });
    }

    let (phase2_outputs, phase2_timings) = judge_with_models_timed(
        client,
        &PHASE_2_MODELS,
        &request.question,
        &request.model_answer,
        &request.correct_answer,
    )
    .await?;
    outputs.extend(phase2_outputs);
    model_timings.extend(phase2_timings);
    let phase2 = &outputs[PHASE_1_MODELS.len()..];
    if phase2[0].verdict == phase2[1].verdict {
        return Ok(UncachedJudgmentResult {
            record: JudgmentCacheRecord {
                key: judgment_cache_key("", &request.split, request.flat_id, &request.model_answer),
                correct_answer: request.correct_answer.clone(),
                is_correct: phase2[0].verdict,
                decision_phase: "phase2_agreement".to_string(),
                judge_outputs: outputs,
                updated_unix_secs: now_unix_secs(),
            },
            model_timings,
        });
    }

    let (output, timing) = judge_with_model_timed(
        client,
        PHASE_3_MODEL,
        &request.question,
        &request.model_answer,
        &request.correct_answer,
    )
    .await?;
    outputs.push(output);
    model_timings.push(timing);
    let final_verdict = outputs.last().unwrap().verdict;
    Ok(UncachedJudgmentResult {
        record: JudgmentCacheRecord {
            key: judgment_cache_key("", &request.split, request.flat_id, &request.model_answer),
            correct_answer: request.correct_answer.clone(),
            is_correct: final_verdict,
            decision_phase: "phase3_final".to_string(),
            judge_outputs: outputs,
            updated_unix_secs: now_unix_secs(),
        },
        model_timings,
    })
}

pub struct CacheChunkLock {
    path: PathBuf,
    stop_heartbeat: Arc<AtomicBool>,
    heartbeat_task: JoinHandle<()>,
}

impl Drop for CacheChunkLock {
    fn drop(&mut self) {
        self.stop_heartbeat.store(true, Ordering::Relaxed);
        self.heartbeat_task.abort();
        let _ = fs::remove_file(&self.path);
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CacheChunkLockMetadata {
    pid: u32,
    hostname: String,
    slurm_job_id: Option<String>,
    acquired_unix_secs: u64,
    heartbeat_unix_secs: u64,
}

fn cache_lock_metadata() -> CacheChunkLockMetadata {
    let now = now_unix_secs();
    CacheChunkLockMetadata {
        pid: std::process::id(),
        hostname: std::env::var("HOSTNAME").unwrap_or_else(|_| "unknown".to_string()),
        slurm_job_id: std::env::var("SLURM_JOB_ID").ok(),
        acquired_unix_secs: now,
        heartbeat_unix_secs: now,
    }
}

fn write_cache_lock_metadata(path: &Path) -> Result<(), String> {
    let metadata = cache_lock_metadata();
    let mut file = File::create(path)
        .map_err(|err| format!("failed to refresh cache lock {}: {err}", path.display()))?;
    serde_json::to_writer(&mut file, &metadata)
        .map_err(|err| format!("failed to serialize cache lock {}: {err}", path.display()))?;
    writeln!(file).map_err(|err| format!("failed to write cache lock {}: {err}", path.display()))
}

fn read_cache_lock_heartbeat(path: &Path) -> Option<u64> {
    let contents = fs::read_to_string(path).ok()?;
    if let Ok(metadata) = serde_json::from_str::<CacheChunkLockMetadata>(&contents) {
        return Some(metadata.heartbeat_unix_secs);
    }
    contents
        .split_whitespace()
        .find_map(|part| part.strip_prefix("time="))
        .and_then(|value| value.parse::<u64>().ok())
}

fn stale_cache_lock_path(lock_path: &Path) -> PathBuf {
    let mut stale_path = lock_path.to_path_buf();
    stale_path.set_extension(format!(
        "jsonl.lock.stale.{}.{}",
        now_unix_secs(),
        std::process::id()
    ));
    stale_path
}

fn start_cache_lock_heartbeat(path: PathBuf, stop_heartbeat: Arc<AtomicBool>) -> JoinHandle<()> {
    tokio::spawn(async move {
        while !stop_heartbeat.load(Ordering::Relaxed) {
            sleep(Duration::from_secs(CACHE_LOCK_HEARTBEAT_SECS)).await;
            if stop_heartbeat.load(Ordering::Relaxed) {
                break;
            }
            if let Err(err) = write_cache_lock_metadata(&path) {
                eprintln!(
                    "failed to refresh judgment cache lock heartbeat {}: {err}",
                    path.display()
                );
            }
        }
    })
}

pub async fn acquire_cache_chunk_lock(
    cache_chunk_file_path: &Path,
) -> Result<CacheChunkLock, String> {
    let lock_path = cache_chunk_file_path.with_extension("jsonl.lock");
    if let Some(parent) = lock_path.parent() {
        fs::create_dir_all(parent).map_err(|err| {
            format!(
                "failed to create cache lock directory {}: {err}",
                parent.display()
            )
        })?;
    }
    loop {
        match OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&lock_path)
        {
            Ok(_file) => {
                write_cache_lock_metadata(&lock_path)?;
                let stop_heartbeat = Arc::new(AtomicBool::new(false));
                let heartbeat_task =
                    start_cache_lock_heartbeat(lock_path.clone(), stop_heartbeat.clone());
                return Ok(CacheChunkLock {
                    path: lock_path,
                    stop_heartbeat,
                    heartbeat_task,
                });
            }
            Err(err) if err.kind() == std::io::ErrorKind::AlreadyExists => {
                let now = now_unix_secs();
                let heartbeat = read_cache_lock_heartbeat(&lock_path);
                if heartbeat
                    .map(|heartbeat| now.saturating_sub(heartbeat) > CACHE_LOCK_STALE_SECS)
                    .unwrap_or(false)
                {
                    let stale_path = stale_cache_lock_path(&lock_path);
                    match fs::rename(&lock_path, &stale_path) {
                        Ok(()) => {
                            eprintln!(
                                "moved stale judgment cache lock {} to {} after >{}s without heartbeat",
                                lock_path.display(),
                                stale_path.display(),
                                CACHE_LOCK_STALE_SECS
                            );
                            continue;
                        }
                        Err(rename_err) if rename_err.kind() == std::io::ErrorKind::NotFound => {
                            continue;
                        }
                        Err(rename_err) => {
                            eprintln!(
                                "failed to move stale judgment cache lock {}: {rename_err}",
                                lock_path.display()
                            );
                        }
                    }
                }
                sleep(Duration::from_secs(CACHE_LOCK_WAIT_SECS)).await;
            }
            Err(err) => {
                return Err(format!(
                    "failed to acquire cache lock {}: {err}",
                    lock_path.display()
                ));
            }
        }
    }
}

pub fn load_cache_chunk(
    path: &Path,
) -> Result<BTreeMap<JudgmentCacheKey, JudgmentCacheRecord>, String> {
    let mut records = BTreeMap::new();
    if !path.exists() {
        return Ok(records);
    }
    let file = File::open(path).map_err(|err| {
        format!(
            "failed to open judgment cache chunk {}: {err}",
            path.display()
        )
    })?;
    for (line_index, line) in BufReader::new(file).lines().enumerate() {
        let line = line.map_err(|err| {
            format!(
                "failed to read judgment cache chunk {} line {}: {err}",
                path.display(),
                line_index + 1
            )
        })?;
        if line.trim().is_empty() {
            continue;
        }
        let record: JudgmentCacheRecord = serde_json::from_str(&line).map_err(|err| {
            format!(
                "failed to parse judgment cache chunk {} line {}: {err}",
                path.display(),
                line_index + 1
            )
        })?;
        records.insert(record.key.clone(), record);
    }
    Ok(records)
}

pub fn rewrite_cache_chunk(
    path: &Path,
    records: &BTreeMap<JudgmentCacheKey, JudgmentCacheRecord>,
) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|err| {
            format!(
                "failed to create judgment cache directory {}: {err}",
                parent.display()
            )
        })?;
    }
    let temp_path = path.with_extension("jsonl.tmp");
    {
        let mut file = File::create(&temp_path).map_err(|err| {
            format!(
                "failed to create judgment cache temp file {}: {err}",
                temp_path.display()
            )
        })?;
        for record in records.values() {
            serde_json::to_writer(&mut file, record).map_err(|err| {
                format!(
                    "failed to serialize judgment cache record to {}: {err}",
                    temp_path.display()
                )
            })?;
            writeln!(file).map_err(|err| {
                format!(
                    "failed to write judgment cache record to {}: {err}",
                    temp_path.display()
                )
            })?;
        }
    }
    fs::rename(&temp_path, path).map_err(|err| {
        format!(
            "failed to replace judgment cache chunk {} with {}: {err}",
            path.display(),
            temp_path.display()
        )
    })
}

pub fn read_judging_requests(path: &Path) -> Result<Vec<JudgingRequestRecord>, String> {
    let file = File::open(path)
        .map_err(|err| format!("failed to open judging input {}: {err}", path.display()))?;
    let mut requests = Vec::new();
    for (line_index, line) in BufReader::new(file).lines().enumerate() {
        let line = line.map_err(|err| {
            format!(
                "failed to read judging input {} line {}: {err}",
                path.display(),
                line_index + 1
            )
        })?;
        if line.trim().is_empty() {
            continue;
        }
        let request = serde_json::from_str::<JudgingRequestRecord>(&line).map_err(|err| {
            format!(
                "failed to parse judging input {} line {}: {err}",
                path.display(),
                line_index + 1
            )
        })?;
        requests.push(request);
    }
    Ok(requests)
}

pub fn read_judging_outputs(path: &Path) -> Result<Vec<JudgingOutputRecord>, String> {
    let file = File::open(path)
        .map_err(|err| format!("failed to open judging output {}: {err}", path.display()))?;
    let mut outputs = Vec::new();
    for (line_index, line) in BufReader::new(file).lines().enumerate() {
        let line = line.map_err(|err| {
            format!(
                "failed to read judging output {} line {}: {err}",
                path.display(),
                line_index + 1
            )
        })?;
        if line.trim().is_empty() {
            continue;
        }
        let output = serde_json::from_str::<JudgingOutputRecord>(&line).map_err(|err| {
            format!(
                "failed to parse judging output {} line {}: {err}",
                path.display(),
                line_index + 1
            )
        })?;
        outputs.push(output);
    }
    Ok(outputs)
}

fn append_escalation_record(
    escalation_jsonl_path: &Path,
    request: &JudgingRequestRecord,
    record: &JudgmentCacheRecord,
) -> Result<(), String> {
    if record.decision_phase != "phase3_final" {
        return Ok(());
    }
    if let Some(parent) = escalation_jsonl_path.parent() {
        fs::create_dir_all(parent).map_err(|err| {
            format!(
                "failed to create escalation directory {}: {err}",
                parent.display()
            )
        })?;
    }
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(escalation_jsonl_path)
        .map_err(|err| {
            format!(
                "failed to open escalation audit log {}: {err}",
                escalation_jsonl_path.display()
            )
        })?;
    let payload = serde_json::json!({
        "request": request,
        "judgment": record,
        "created_unix_secs": now_unix_secs(),
    });
    serde_json::to_writer(&mut file, &payload).map_err(|err| {
        format!(
            "failed to serialize escalation audit record to {}: {err}",
            escalation_jsonl_path.display()
        )
    })?;
    writeln!(file).map_err(|err| {
        format!(
            "failed to write escalation audit record to {}: {err}",
            escalation_jsonl_path.display()
        )
    })
}

pub async fn judge_requests(
    requests: Vec<JudgingRequestRecord>,
    output_jsonl_path: &Path,
    cache_dir: &Path,
    escalation_jsonl_path: &Path,
    cache_version: &str,
    cache_chunk_question_count: usize,
    request_concurrency_per_model: usize,
) -> Result<JudgingSummary, String> {
    let job_start = Instant::now();
    let client = Client::new();
    let semaphore = std::sync::Arc::new(Semaphore::new(request_concurrency_per_model));
    let mut requests_by_cache_path: BTreeMap<PathBuf, Vec<JudgingRequestRecord>> = BTreeMap::new();
    for request in requests {
        let cache_path = cache_chunk_path(
            cache_dir,
            &request.split,
            request.flat_id,
            cache_chunk_question_count,
        );
        requests_by_cache_path
            .entry(cache_path)
            .or_default()
            .push(request);
    }

    if let Some(parent) = output_jsonl_path.parent() {
        fs::create_dir_all(parent).map_err(|err| {
            format!(
                "failed to create judging output directory {}: {err}",
                parent.display()
            )
        })?;
    }
    let temp_output_path = output_jsonl_path.with_extension("jsonl.tmp");
    let mut output_file = File::create(&temp_output_path).map_err(|err| {
        format!(
            "failed to create judging output {}: {err}",
            temp_output_path.display()
        )
    })?;
    let mut summary = JudgingSummary::default();

    for (cache_path, chunk_requests) in requests_by_cache_path {
        summary.cache_chunks_processed += 1;
        let _lock = acquire_cache_chunk_lock(&cache_path).await?;
        let mut cache_records = load_cache_chunk(&cache_path)?;
        let mut outputs_by_request_index: BTreeMap<usize, JudgingOutputRecord> = BTreeMap::new();
        let mut uncached_by_key: BTreeMap<JudgmentCacheKey, Vec<(usize, JudgingRequestRecord)>> =
            BTreeMap::new();
        for (index, request) in chunk_requests.iter().enumerate() {
            summary.total_requests += 1;
            let key = judgment_cache_key(
                cache_version,
                &request.split,
                request.flat_id,
                &request.model_answer,
            );
            if let Some(record) = cache_records.get(&key) {
                summary.cache_hits += 1;
                outputs_by_request_index.insert(
                    index,
                    JudgingOutputRecord {
                        request: request.clone(),
                        is_correct: record.is_correct,
                        decision_phase: record.decision_phase.clone(),
                        cache_hit: true,
                    },
                );
            } else {
                uncached_by_key
                    .entry(key)
                    .or_default()
                    .push((index, request.clone()));
            }
        }

        let mut judged_results =
            stream::iter(uncached_by_key.into_iter().map(|(key, requests)| {
                let client = client.clone();
                let semaphore = semaphore.clone();
                async move {
                    let representative_request = requests
                        .first()
                        .expect("uncached key must have at least one request")
                        .1
                        .clone();
                    let _permit = semaphore
                        .acquire_owned()
                        .await
                        .map_err(|err| format!("judging semaphore closed: {err}"))?;
                    let mut judgment_result =
                        judge_uncached_request(&client, &representative_request).await?;
                    judgment_result.record.key = key;
                    Ok::<_, String>((requests, representative_request, judgment_result))
                }
            }))
            .buffer_unordered(request_concurrency_per_model);

        let mut successful_judgments_since_flush = 0usize;
        while let Some(result) = judged_results.next().await {
            let (requests, representative_request, judgment_result) = match result {
                Ok(result) => result,
                Err(error) => {
                    if successful_judgments_since_flush > 0 {
                        rewrite_cache_chunk(&cache_path, &cache_records)?;
                    }
                    eprintln!(
                        "judging_failed=1 cache_path={} reason={}",
                        cache_path.display(),
                        error
                    );
                    return Err(error);
                }
            };
            let record = judgment_result.record;
            for timing in judgment_result.model_timings {
                summary.record_model_timing(timing);
            }
            summary.newly_judged += 1;
            match record.decision_phase.as_str() {
                "phase1_unanimous" => summary.phase1_unanimous += 1,
                "phase2_agreement" => summary.phase2_agreement += 1,
                "phase3_final" => summary.phase3_escalations += 1,
                "exact_match" => summary.exact_matches += 1,
                "empty_answer" => summary.failures_marked_incorrect += 1,
                _ => {}
            }
            append_escalation_record(escalation_jsonl_path, &representative_request, &record)?;
            for (index, request) in requests {
                outputs_by_request_index.insert(
                    index,
                    JudgingOutputRecord {
                        request,
                        is_correct: record.is_correct,
                        decision_phase: record.decision_phase.clone(),
                        cache_hit: false,
                    },
                );
            }
            cache_records.insert(record.key.clone(), record);
            successful_judgments_since_flush += 1;
            if successful_judgments_since_flush >= CACHE_FLUSH_INTERVAL {
                rewrite_cache_chunk(&cache_path, &cache_records)?;
                successful_judgments_since_flush = 0;
            }
        }

        if successful_judgments_since_flush > 0 {
            rewrite_cache_chunk(&cache_path, &cache_records)?;
        }
        for output in outputs_by_request_index.values() {
            serde_json::to_writer(&mut output_file, output).map_err(|err| {
                format!(
                    "failed to serialize judging output to {}: {err}",
                    temp_output_path.display()
                )
            })?;
            writeln!(output_file).map_err(|err| {
                format!(
                    "failed to write judging output to {}: {err}",
                    temp_output_path.display()
                )
            })?;
        }
    }

    fs::rename(&temp_output_path, output_jsonl_path).map_err(|err| {
        format!(
            "failed to replace judging output {} with {}: {err}",
            output_jsonl_path.display(),
            temp_output_path.display()
        )
    })?;
    summary.finalize_rates(job_start.elapsed().as_secs_f64());
    Ok(summary)
}
