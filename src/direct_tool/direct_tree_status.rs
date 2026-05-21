use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum DirectTreeStatus {
    CreatingTrunkTrajectory,
    CreatingOrChoosingBranchPoint,
    CreatingBranchSegment,
    // JudgingBranchSegment,
    Complete,
}
