use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum DirectTreeStatus {
    CreatingTrunkTrajectory,
    Complete,
    // guided branching mode
    CreatingOrChoosingBranchPoint,
    CreatingBranchSegment,
    // spontaneous branching mode
    SpontaneousBranching,
}
