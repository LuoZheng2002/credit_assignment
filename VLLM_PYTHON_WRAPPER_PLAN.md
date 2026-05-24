This project requires accessing vLLM, and we want the following for inputs and outputs:
1. For the requests, we want to send token_ids directly (that have chat template already applied) instead of prompt texts.
2. For the response, we want the logprobs with tokens in the form of token_ids

These requirements cannot be fulfilled by a vllm server through `vllm serve`, because the API does not accept token ids as input and does not provide token ids as output.

Therefore, we'd like to create a python wrapper that interacts with vllm through the python API, so that the token id input and output functionality can be preserved.

The consumer side is written in Rust, so we need inter-process communication.

We plan to use gRPC (protobuf over HTTP/2) to communicate between Python and Rust.

The contract should be similar to the definitions in src/vllm_wrapper.rs.

After writing the python server code, we need to implement another backend called "vllm-wrapper" for src/llm_model/qwen_shared.rs.

Currently the Rust API to connect with is LlmCallable::generate_tokens_with_logprobs in src/llm_model/mod.rs.

---

## Pre-Implementation Decision Checklist

Use this checklist to lock decisions before coding. Keep all final choices in this file.

### 1) IPC + Protocol (gRPC)

- [x] **IPC transport**
  - Decision: tcp_localhost
  - Notes: Maximize compatibility

- [x] **Protocol family**
  - Decision: gRPC (protobuf RPC service)
  - Notes: supersedes custom framing decision

- [x] **gRPC method style**
  - Decision: unary generate
  - Notes: `<initial scope and future extension path>`

- [x] **gRPC endpoint format**
  - Decision: 127.0.0.1:50051
  - Notes: Like vLLM, only the port number is passed to Rust CLI args, assuming server and client are in the same compute node

- [x] **gRPC channel security mode**
  - Decision: insecure_localhost_only
  - Notes: `<threat model and deployment environment>`

- [x] **gRPC channel options**
  - Decision: Use defaults. `<max message size, keepalive, compression, connection reuse>`
  - Notes: `<defaults and rationale>`

- [x] **Proto schema versioning strategy**
  - Decision: package vllm_wrapper.v1 + backward-compatible field additions
  - Notes: We typically update both Python and Rust side, and it is cheap to do so in this project.

### 2) API Contract (Request / Response)

- [x] **Prompt input shape**
  - Decision: oneof{text, token_ids}
  - Notes: We assume the text prompt also has chat template already applied

- [x] **Response token semantics**
  - Decision: generated_tokens_only
  - Notes: `<must be unambiguous for Rust consumer>`

- [x] **Logprobs shape and guarantees**
  - Decision:
    - top-k = always 8
    - guarantee `len(logprobs) == len(tokens)` = yes
    - padding rule when < k candidates = fill with token_id=sampled, logprob=-inf
  - Notes: `<null handling and deterministic ordering rules>`

- [x] **Decoded text in response**
  - Decision: always returned by Python side, may not be used by Rust side
  - Notes: the logprob tokens are the source of truth for generated content if logprob is required

### 3) Stop / Sampling / Generation Behavior

- [x] **Stop condition type**
  - Decision: stop_strings
  - Notes: the stop strings should be contained in the output if the model has generated them

- [x] **Sampling parameter parity with existing Rust fields**
  - Decision: final supported fields: max_tokens, temperature, stop, include_stop_str_in_output, requires_logprobs
  - Notes: `<defaults and validation rules>`

- [x] **Determinism controls**
  - Decision: No deterministic control is required for llm generation
  - Notes: `<reproducibility requirements for experiments>`

### 4) Errors, Timeouts, Retries

- [x] **Typed error model**
  - Decision: Use strings to express error, no typed error needed
  - Notes: `<mapping from Python/vLLM exceptions to codes>`

- [x] **gRPC status mapping**
  - Decision: Turn gRPC status codes to error string if error occurs
  - Notes: `<e.g., INVALID_ARGUMENT, DEADLINE_EXCEEDED, UNAVAILABLE, INTERNAL>`

- [x] **Timeout policy**
  - Decision:
    - request timeout = Set to very large number / no timeout
    - connect/startup timeout = Set to very large number / no timeout
  - Notes: `<where enforced: Rust client, Python server, or both>`

- [x] **Retry policy**
  - Decision: Retry 3 times, if none of them succeed, exit the program with error message. For now, retry for all errors. If the server does not respond, wait indefinitely.
  - Notes: `<idempotency assumptions>`

### 5) Rust Integration

- [x] **New backend wiring in Rust**
  - Decision: backend enum name: VllmWrapper, cli name:vllm-wrapper, cli args: --vllm-wrapper-port, --model-cli-name (of type LlmModelName)
  - Notes: `<how this coexists with current qwen_api_backend=vllm/openrouter>`

- [x] **Contract alignment target**
  - Decision: The source of truth is the proto file, and the Rust side should wrap the generated code with custom struct.
  - Notes: prefer 32 bit types, but the Rust side automatic code generation from proto may use 64 bit types anyway.

- [x] **Code generation toolchain**
  - Decision: Rust: tonic+prost, Python: grpcio-tools
  - Notes: `<how generated code is versioned and rebuilt>`

- [x] **Concurrency ownership**
  - Decision: Rust semaphore only
  - Notes: `<limits and rationale>`

### 6) Python Wrapper Runtime

- [x] **Model lifecycle**
  - Decision: single model per process
  - Notes: `<startup latency and memory implications>`

- [x] **Health/readiness contract**
  - Decision: Rust side waits indefinitely for the server side to go online.
  - Notes: `<what Rust checks before sending requests>`

- [x] **Tokenizer compatibility checks**
  - Decision: no
  - Notes: `<behavior on mismatch between Rust token_ids and Python tokenizer/model>`

### 7) Observability + Validation

- [x] **Structured logging fields**
  - Decision: request_id, model_name, prompt_token_count, generated_token_count, latency_ms, error_code
  - Notes: `<PII/redaction expectations>`

- [x] **Metrics to emit**
  - Decision: success rate, timeout rate, tokens/sec, p50/p95 latency
  - Notes: `<where collected>`

- [x] **Acceptance tests before merge**
  - Decision: No test required before all the functionalities are implemented
  - Suggested minimum:
    - token_ids request round-trip
    - token_id logprobs shape/length/top-k correctness
    - stop behavior correctness
    - context-length error mapping
    - concurrency stress smoke test

---

## Final Sign-off

- [x] All checklist items above decided
- [x] Proto contract frozen for first implementation
- [x] Rust + Python owners agree on integration surface
