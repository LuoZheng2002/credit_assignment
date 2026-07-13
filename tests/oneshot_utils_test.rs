use credit_assignment::oneshot_utils::{
    OneshotRunManifest, all_expected_oneshot_model_outputs_exist, detect_oneshot_artifacts,
    read_oneshot_run_manifest, write_oneshot_run_manifest,
};

#[test]
fn oneshot_manifest_roundtrips() {
    let temp_dir = tempfile::tempdir().unwrap();
    let summary_parent_dir = temp_dir.path().join("summary");
    let summary_parent_dir_str = summary_parent_dir.to_string_lossy().into_owned();

    write_oneshot_run_manifest(&summary_parent_dir_str, 7);
    let manifest = read_oneshot_run_manifest(&summary_parent_dir_str);
    assert_eq!(
        manifest,
        Some(OneshotRunManifest {
            num_oneshot_epochs: 7
        })
    );
}

#[test]
fn detect_oneshot_artifacts_notices_epoch_output_dirs() {
    let temp_dir = tempfile::tempdir().unwrap();
    let summary_parent_dir = temp_dir.path().join("summary");
    let model_output_root = temp_dir.path().join("models");
    std::fs::create_dir_all(model_output_root.join("oneshot_epoch_1/model")).unwrap();

    assert!(detect_oneshot_artifacts(
        &summary_parent_dir.to_string_lossy(),
        &model_output_root.to_string_lossy(),
    ));
}

#[test]
fn expected_oneshot_outputs_require_all_epochs() {
    let temp_dir = tempfile::tempdir().unwrap();
    let model_output_root = temp_dir.path().join("models");
    std::fs::create_dir_all(model_output_root.join("oneshot_epoch_1/model")).unwrap();
    std::fs::create_dir_all(model_output_root.join("oneshot_epoch_2/model")).unwrap();

    assert!(all_expected_oneshot_model_outputs_exist(
        &model_output_root.to_string_lossy(),
        2,
    ));
    assert!(!all_expected_oneshot_model_outputs_exist(
        &model_output_root.to_string_lossy(),
        3,
    ));
}
