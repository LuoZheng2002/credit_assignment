pub const FORCED_END_MESSAGE: &str =
    "The model does not manage to provide a final answer within allowed number of turns.";
pub const CONTEXT_LENGTH_EXCEEDED_ABORT_MESSAGE: &str =
    "<error>Model context length exceeded, aborting.</error>";
pub const IDENTICAL_PYTHON_ERROR_ABORT_MESSAGE: &str =
    "<error>Identical python tool error detected. Aborting current incomplete step.</error>";
pub const REPETITION_ABORT_MESSAGE: &str =
    "<error>Repeated contents detected. This step is forced to abort without completion.</error>";

pub const MAX_PLAN_CHANGES: usize = 2;
pub const MAX_STEP_OVERWRITE_STREAK: usize = 2;
pub const MAX_TOTAL_STEP_OVERWRITES: usize = 6;
pub const MAX_ACTIONS_PER_STEP: usize = 30;
