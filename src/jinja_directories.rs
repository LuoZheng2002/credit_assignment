use serde::Serialize;
use serde_json::json;
use std::sync::LazyLock;

use crate::direct_tool::hybrid_dataset::DatasetSplit;
use crate::utils::{load_jinja_template_environment, mount_dir};

const ACTION_LOGS_TRAINING_PATH_EXTSORT_TEMPLATE_PATH: &str =
    "config/directories/action_logs_training_path_extsort.jinja";
const ACTION_LOGS_VALIDATION_PATH_EXTSORT_TEMPLATE_PATH: &str =
    "config/directories/action_logs_validation_path_extsort.jinja";
const ACTION_LOGS_TESTING_PATH_EXTSORT_TEMPLATE_PATH: &str =
    "config/directories/action_logs_testing_path_extsort.jinja";
const INFERENCE_WRAPPER_LOG_PATH_TEMPLATE_PATH: &str =
    "config/directories/inference_wrapper_log_path.jinja";
const MODEL_PARENT_DIR_TEMPLATE_PATH: &str = "config/directories/model_parent_dir.jinja";
const MODEL_CHECKPOINT_DIR_TEMPLATE_PATH: &str = "config/directories/model_checkpoint_dir.jinja";
const MODEL_METRICS_PATH_TEMPLATE_PATH: &str = "config/directories/model_metrics_path.jinja";
const PROGRESS_SAVE_PATH_TEMPLATE_PATH: &str = "config/directories/progress_save_path.jinja";
const TEST_ACCURACY_PATH_TEMPLATE_PATH: &str = "config/directories/test_accuracy_path.jinja";
const TRAINING_TRAJECTORIES_PATH_TEMPLATE_PATH: &str =
    "config/directories/training_trajectories_msgpack_path.jinja";
const TRAINING_TRAJECTORIES_STATS_PATH_TEMPLATE_PATH: &str =
    "config/directories/training_trajectories_stats_msgpack_path.jinja";
const TRAINING_SUMMARY_PARENT_DIR_TEMPLATE_PATH: &str =
    "config/directories/training_summary_parent_dir.jinja";
const TRAINING_WRAPPER_LOG_PATH_TEMPLATE_PATH: &str =
    "config/directories/training_wrapper_log_path.jinja";
const SFT_WRAPPER_LOG_PATH_TEMPLATE_PATH: &str = "config/directories/sft_wrapper_log_path.jinja";
const SFT_MODEL_PARENT_DIR_TEMPLATE_PATH: &str = "config/directories/sft_model_parent_dir.jinja";
const TUI_LOG_PATH_TEMPLATE_PATH: &str = "config/directories/tui_log_path.jinja";

static ACTION_LOGS_TRAINING_PATH_EXTSORT_TEMPLATE_ENVIRONMENT: LazyLock<
    Result<minijinja::Environment<'static>, String>,
> = LazyLock::new(|| {
    load_jinja_template_environment(
        ACTION_LOGS_TRAINING_PATH_EXTSORT_TEMPLATE_PATH,
        "action_logs_training_path_extsort",
    )
});

static ACTION_LOGS_VALIDATION_PATH_EXTSORT_TEMPLATE_ENVIRONMENT: LazyLock<
    Result<minijinja::Environment<'static>, String>,
> = LazyLock::new(|| {
    load_jinja_template_environment(
        ACTION_LOGS_VALIDATION_PATH_EXTSORT_TEMPLATE_PATH,
        "action_logs_validation_path_extsort",
    )
});

static ACTION_LOGS_TESTING_PATH_EXTSORT_TEMPLATE_ENVIRONMENT: LazyLock<
    Result<minijinja::Environment<'static>, String>,
> = LazyLock::new(|| {
    load_jinja_template_environment(
        ACTION_LOGS_TESTING_PATH_EXTSORT_TEMPLATE_PATH,
        "action_logs_testing_path_extsort",
    )
});

static INFERENCE_WRAPPER_LOG_PATH_TEMPLATE_ENVIRONMENT: LazyLock<
    Result<minijinja::Environment<'static>, String>,
> = LazyLock::new(|| {
    load_jinja_template_environment(
        INFERENCE_WRAPPER_LOG_PATH_TEMPLATE_PATH,
        "inference_wrapper_log_path",
    )
});

static MODEL_PARENT_DIR_TEMPLATE_ENVIRONMENT: LazyLock<
    Result<minijinja::Environment<'static>, String>,
> = LazyLock::new(|| {
    load_jinja_template_environment(MODEL_PARENT_DIR_TEMPLATE_PATH, "model_parent_dir")
});

static MODEL_CHECKPOINT_DIR_TEMPLATE_ENVIRONMENT: LazyLock<
    Result<minijinja::Environment<'static>, String>,
> = LazyLock::new(|| {
    load_jinja_template_environment(MODEL_CHECKPOINT_DIR_TEMPLATE_PATH, "model_checkpoint_dir")
});

static MODEL_METRICS_PATH_TEMPLATE_ENVIRONMENT: LazyLock<
    Result<minijinja::Environment<'static>, String>,
> = LazyLock::new(|| {
    load_jinja_template_environment(MODEL_METRICS_PATH_TEMPLATE_PATH, "model_metrics_path")
});

static PROGRESS_SAVE_PATH_TEMPLATE_ENVIRONMENT: LazyLock<
    Result<minijinja::Environment<'static>, String>,
> = LazyLock::new(|| {
    load_jinja_template_environment(PROGRESS_SAVE_PATH_TEMPLATE_PATH, "progress_save_path")
});

static TEST_ACCURACY_PATH_TEMPLATE_ENVIRONMENT: LazyLock<
    Result<minijinja::Environment<'static>, String>,
> = LazyLock::new(|| {
    load_jinja_template_environment(TEST_ACCURACY_PATH_TEMPLATE_PATH, "test_accuracy_path")
});

static TRAINING_TRAJECTORIES_PATH_TEMPLATE_ENVIRONMENT: LazyLock<
    Result<minijinja::Environment<'static>, String>,
> = LazyLock::new(|| {
    load_jinja_template_environment(
        TRAINING_TRAJECTORIES_PATH_TEMPLATE_PATH,
        "training_trajectories_path",
    )
});

static TRAINING_TRAJECTORIES_STATS_PATH_TEMPLATE_ENVIRONMENT: LazyLock<
    Result<minijinja::Environment<'static>, String>,
> = LazyLock::new(|| {
    load_jinja_template_environment(
        TRAINING_TRAJECTORIES_STATS_PATH_TEMPLATE_PATH,
        "training_trajectories_stats_path",
    )
});

static TRAINING_SUMMARY_PARENT_DIR_TEMPLATE_ENVIRONMENT: LazyLock<
    Result<minijinja::Environment<'static>, String>,
> = LazyLock::new(|| {
    load_jinja_template_environment(
        TRAINING_SUMMARY_PARENT_DIR_TEMPLATE_PATH,
        "training_summary_parent_dir",
    )
});

static TRAINING_WRAPPER_LOG_PATH_TEMPLATE_ENVIRONMENT: LazyLock<
    Result<minijinja::Environment<'static>, String>,
> = LazyLock::new(|| {
    load_jinja_template_environment(
        TRAINING_WRAPPER_LOG_PATH_TEMPLATE_PATH,
        "training_wrapper_log_path",
    )
});

static SFT_WRAPPER_LOG_PATH_TEMPLATE_ENVIRONMENT: LazyLock<
    Result<minijinja::Environment<'static>, String>,
> = LazyLock::new(|| {
    load_jinja_template_environment(SFT_WRAPPER_LOG_PATH_TEMPLATE_PATH, "sft_wrapper_log_path")
});

static SFT_MODEL_PARENT_DIR_TEMPLATE_ENVIRONMENT: LazyLock<
    Result<minijinja::Environment<'static>, String>,
> = LazyLock::new(|| {
    load_jinja_template_environment(SFT_MODEL_PARENT_DIR_TEMPLATE_PATH, "sft_model_parent_dir")
});

static TUI_LOG_PATH_TEMPLATE_ENVIRONMENT: LazyLock<
    Result<minijinja::Environment<'static>, String>,
> = LazyLock::new(|| load_jinja_template_environment(TUI_LOG_PATH_TEMPLATE_PATH, "tui_log_path"));

fn render_template(
    template_env: &LazyLock<Result<minijinja::Environment<'static>, String>>,
    template_name: &'static str,
    context: impl Serialize,
) -> Result<String, String> {
    let env = template_env.as_ref().map_err(|err| err.clone())?;
    let template = env
        .get_template(template_name)
        .map_err(|err| format!("Failed to load {} template: {}", template_name, err))?;
    let rendered = template
        .render(context)
        .map_err(|err| format!("Failed to render {} template: {}", template_name, err))?;
    let rendered = rendered.trim().to_string();
    if rendered.is_empty() {
        return Err(format!("Rendered {} template is empty", template_name));
    }
    Ok(rendered)
}

fn render_epoch_template(
    template_env: &LazyLock<Result<minijinja::Environment<'static>, String>>,
    template_name: &'static str,
    model_cli_name: &str,
    config_nickname: &str,
    epoch: usize,
) -> Result<String, String> {
    let mount_dir = mount_dir()?;
    render_template(
        template_env,
        template_name,
        json!({
            "mount_dir": mount_dir,
            "model_cli_name": model_cli_name,
            "config_nickname": config_nickname,
            "epoch": epoch,
        }),
    )
}

fn render_model_config_template(
    template_env: &LazyLock<Result<minijinja::Environment<'static>, String>>,
    template_name: &'static str,
    model_cli_name: &str,
    config_nickname: &str,
) -> Result<String, String> {
    let mount_dir = mount_dir()?;
    render_template(
        template_env,
        template_name,
        json!({
            "mount_dir": mount_dir,
            "model_cli_name": model_cli_name,
            "config_nickname": config_nickname,
        }),
    )
}

fn render_training_trajectories_template(
    template_env: &LazyLock<Result<minijinja::Environment<'static>, String>>,
    template_name: &'static str,
    model_cli_name: &str,
    config_nickname: &str,
    epoch: usize,
) -> Result<String, String> {
    let mount_dir = mount_dir()?;
    render_template(
        template_env,
        template_name,
        json!({
            "mount_dir": mount_dir,
            "model_cli_name": model_cli_name,
            "config_nickname": config_nickname,
            "epoch": epoch,
        }),
    )
}

pub fn action_logs_path_from_template<S: DatasetSplit>(
    model_cli_name: &str,
    config_nickname: &str,
    epoch: usize,
) -> Result<String, String> {
    match S::dataset_file_postfix().as_str() {
        "train" => render_epoch_template(
            &ACTION_LOGS_TRAINING_PATH_EXTSORT_TEMPLATE_ENVIRONMENT,
            "action_logs_training_path_extsort",
            model_cli_name,
            config_nickname,
            epoch,
        ),
        "val" => render_epoch_template(
            &ACTION_LOGS_VALIDATION_PATH_EXTSORT_TEMPLATE_ENVIRONMENT,
            "action_logs_validation_path_extsort",
            model_cli_name,
            config_nickname,
            epoch,
        ),
        "test" => render_epoch_template(
            &ACTION_LOGS_TESTING_PATH_EXTSORT_TEMPLATE_ENVIRONMENT,
            "action_logs_testing_path_extsort",
            model_cli_name,
            config_nickname,
            epoch,
        ),
        other => Err(format!("Unsupported dataset split postfix: {}", other)),
    }
}

pub fn inference_wrapper_log_path_from_template(
    model_cli_name: &str,
    config_nickname: &str,
) -> Result<String, String> {
    render_model_config_template(
        &INFERENCE_WRAPPER_LOG_PATH_TEMPLATE_ENVIRONMENT,
        "inference_wrapper_log_path",
        model_cli_name,
        config_nickname,
    )
}

pub fn model_parent_dir_from_template(
    model_cli_name: &str,
    config_nickname: &str,
    epoch: usize,
) -> Result<String, String> {
    render_epoch_template(
        &MODEL_PARENT_DIR_TEMPLATE_ENVIRONMENT,
        "model_parent_dir",
        model_cli_name,
        config_nickname,
        epoch,
    )
}

pub fn model_checkpoint_dir_from_template(
    model_cli_name: &str,
    config_nickname: &str,
    epoch: usize,
) -> Result<String, String> {
    render_epoch_template(
        &MODEL_CHECKPOINT_DIR_TEMPLATE_ENVIRONMENT,
        "model_checkpoint_dir",
        model_cli_name,
        config_nickname,
        epoch,
    )
}

pub fn model_metrics_path_from_template(
    model_cli_name: &str,
    config_nickname: &str,
    epoch: usize,
) -> Result<String, String> {
    render_epoch_template(
        &MODEL_METRICS_PATH_TEMPLATE_ENVIRONMENT,
        "model_metrics_path",
        model_cli_name,
        config_nickname,
        epoch,
    )
}

pub fn progress_save_path_from_template(
    model_cli_name: &str,
    config_nickname: &str,
) -> Result<String, String> {
    render_model_config_template(
        &PROGRESS_SAVE_PATH_TEMPLATE_ENVIRONMENT,
        "progress_save_path",
        model_cli_name,
        config_nickname,
    )
}

pub fn test_accuracy_path_from_template(
    model_cli_name: &str,
    config_nickname: &str,
    epoch: usize,
) -> Result<String, String> {
    render_epoch_template(
        &TEST_ACCURACY_PATH_TEMPLATE_ENVIRONMENT,
        "test_accuracy_path",
        model_cli_name,
        config_nickname,
        epoch,
    )
}

pub fn training_trajectories_path_from_template(
    model_cli_name: &str,
    config_nickname: &str,
    epoch: usize,
) -> Result<String, String> {
    render_training_trajectories_template(
        &TRAINING_TRAJECTORIES_PATH_TEMPLATE_ENVIRONMENT,
        "training_trajectories_path",
        model_cli_name,
        config_nickname,
        epoch,
    )
}

pub fn training_trajectories_stats_path_from_template(
    model_cli_name: &str,
    config_nickname: &str,
    epoch: usize,
) -> Result<String, String> {
    render_training_trajectories_template(
        &TRAINING_TRAJECTORIES_STATS_PATH_TEMPLATE_ENVIRONMENT,
        "training_trajectories_stats_path",
        model_cli_name,
        config_nickname,
        epoch,
    )
}

pub fn training_summary_parent_dir_from_template(
    model_cli_name: &str,
    config_nickname: &str,
    epoch: usize,
) -> Result<String, String> {
    render_epoch_template(
        &TRAINING_SUMMARY_PARENT_DIR_TEMPLATE_ENVIRONMENT,
        "training_summary_parent_dir",
        model_cli_name,
        config_nickname,
        epoch,
    )
}

pub fn training_wrapper_log_path_from_template(
    model_cli_name: &str,
    config_nickname: &str,
) -> Result<String, String> {
    render_model_config_template(
        &TRAINING_WRAPPER_LOG_PATH_TEMPLATE_ENVIRONMENT,
        "training_wrapper_log_path",
        model_cli_name,
        config_nickname,
    )
}

pub fn sft_wrapper_log_path_from_template(
    model_cli_name: &str,
    config_nickname: &str,
) -> Result<String, String> {
    render_model_config_template(
        &SFT_WRAPPER_LOG_PATH_TEMPLATE_ENVIRONMENT,
        "sft_wrapper_log_path",
        model_cli_name,
        config_nickname,
    )
}

pub fn sft_model_parent_dir_from_template(
    model_cli_name: &str,
    config_nickname: &str,
) -> Result<String, String> {
    render_model_config_template(
        &SFT_MODEL_PARENT_DIR_TEMPLATE_ENVIRONMENT,
        "sft_model_parent_dir",
        model_cli_name,
        config_nickname,
    )
}

pub fn tui_log_path_from_template(
    model_cli_name: &str,
    config_nickname: &str,
) -> Result<String, String> {
    render_model_config_template(
        &TUI_LOG_PATH_TEMPLATE_ENVIRONMENT,
        "tui_log_path",
        model_cli_name,
        config_nickname,
    )
}
