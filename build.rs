fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=proto/vllm_wrapper.proto");

    let protoc = protoc_bin_vendored::protoc_bin_path()?;

    let mut config = tonic_prost_build::Config::new();
    config.protoc_executable(protoc);

    tonic_prost_build::configure()
        .build_server(false)
        .compile_with_config(config, &["proto/vllm_wrapper.proto"], &["proto"])?;

    Ok(())
}
