use std::collections::BTreeSet;
use std::fs::File;
use std::path::PathBuf;

use clap::Parser;
use credit_assignment::agent::tree_schema::CompletedTree;
use credit_assignment::em::em_dataset_builder::EmDatasetBuilder;
use credit_assignment::em::em_fitting::EmFitter;
use credit_assignment::em::em_schema::split_em_fit_result_per_tree;
use credit_assignment::em::em_types::{
    EmFitDiagnostics, EmGlobalConfigSnapshot, EmHyperparameters, EmNodeTypePosterior, LogStdClamp,
};
use credit_assignment::parallel_process_jsonl::{read_json_lines, write_jsonl_file};
use serde::Serialize;

#[derive(Serialize, Debug)]
struct EmFitMetaOutput {
    global: Vec<EmNodeTypePosterior>,
    config: EmGlobalConfigSnapshot,
    diagnostics: EmFitDiagnostics,
}

#[derive(Parser, Debug)]
#[command(name = "Fit EM over trees")]
struct Args {
    #[arg(long, help = "Input jsonl path of CompletedTree entries")]
    trees_file: PathBuf,

    #[arg(long, help = "Output jsonl path for EmFitPerTree entries")]
    output_file: PathBuf,

    #[arg(
        long,
        help = "Output json path for EM fit metadata (global priors/config/diagnostics)"
    )]
    em_fit_meta_file: PathBuf,

    #[arg(long, default_value_t = 1.0)]
    sigma_ordinary: f64,

    #[arg(long, default_value_t = 1.0)]
    sigma_special: f64,

    #[arg(long, default_value_t = 1.0)]
    sigma_log_std: f64,

    #[arg(long, default_value_t = 1.0)]
    lambda_slack: f64,

    #[arg(long, default_value_t = 1e-6)]
    eps: f64,

    #[arg(long, default_value_t = 100)]
    max_iterations: usize,

    #[arg(long, default_value_t = -4.0)]
    log_std_min: f64,

    #[arg(long, default_value_t = 2.0)]
    log_std_max: f64,
}

fn main() {
    let args = Args::parse();
    let completed_trees: Vec<CompletedTree> =
        read_json_lines(&args.trees_file).unwrap_or_else(|err| {
            panic!(
                "Failed to read trees file {}: {}",
                args.trees_file.display(),
                err
            )
        });
    assert!(
        !completed_trees.is_empty(),
        "Input trees file must contain at least one CompletedTree entry"
    );

    let mut expected_tree_ids: BTreeSet<usize> = BTreeSet::new();
    let mut fit_trees = Vec::with_capacity(completed_trees.len());
    for completed_tree in &completed_trees {
        assert_eq!(
            completed_tree.id, completed_tree.trajectory.question_id,
            "CompletedTree.id must equal Tree.question_id"
        );
        assert!(
            expected_tree_ids.insert(completed_tree.id),
            "Duplicate CompletedTree.id found in input: {}",
            completed_tree.id
        );
        fit_trees.push(completed_tree.trajectory.clone());
    }

    let dataset = EmDatasetBuilder::new().build_from_trees(&fit_trees);
    let fitter = EmFitter::new(EmHyperparameters {
        sigma_ordinary: args.sigma_ordinary,
        sigma_special: args.sigma_special,
        sigma_log_std: args.sigma_log_std,
        lambda_slack: args.lambda_slack,
        eps: args.eps,
        max_iterations: args.max_iterations,
        log_std_clamp: LogStdClamp {
            min: args.log_std_min,
            max: args.log_std_max,
        },
    });
    let fit_result = fitter.fit(&dataset);
    let per_tree = split_em_fit_result_per_tree(&fit_result);

    assert_eq!(
        per_tree.len(),
        completed_trees.len(),
        "Output EmFitPerTree entries must match input CompletedTree count"
    );
    let actual_tree_ids: BTreeSet<usize> = per_tree
        .iter()
        .map(|entry| entry.tree_question_id)
        .collect();
    assert_eq!(
        actual_tree_ids, expected_tree_ids,
        "Output EmFitPerTree tree id set must match input CompletedTree id set"
    );

    write_jsonl_file(&args.output_file, &per_tree).unwrap_or_else(|err| {
        panic!(
            "Failed to write EmFitPerTree output to {}: {}",
            args.output_file.display(),
            err
        )
    });
    let meta_output = EmFitMetaOutput {
        global: fit_result.global,
        config: fit_result.config,
        diagnostics: fit_result.diagnostics,
    };
    let meta_file = File::create(&args.em_fit_meta_file).unwrap_or_else(|err| {
        panic!(
            "Failed to create EM fit metadata output file {}: {}",
            args.em_fit_meta_file.display(),
            err
        )
    });
    serde_json::to_writer_pretty(meta_file, &meta_output).unwrap_or_else(|err| {
        panic!(
            "Failed to write EM fit metadata output to {}: {}",
            args.em_fit_meta_file.display(),
            err
        )
    });
    println!(
        "Wrote {} EmFitPerTree entries to {}",
        per_tree.len(),
        args.output_file.display()
    );
    println!(
        "Wrote EM fit metadata to {}",
        args.em_fit_meta_file.display()
    );
}
