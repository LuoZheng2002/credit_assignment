# Modal Support Plan (Concrete)

## Goal

Add a new **Modal backend** for both inference and training while keeping current **HPC/local backend** behavior unchanged.

Current pipeline in `src/orchestrator.rs` remains conceptually the same:
1. Collect rollouts (inference calls)
2. Process/filter trajectories in Rust
3. Train model on processed trajectories

The backend switch should change only _where_ inference/training run, not the orchestration logic.

---

## Current Code Anchors (What We Reuse)

- Inference server lifecycle is currently owned by:
  - `src/orchestrator.rs` (`ensure_inference_server_launched`, `launch_inference_server`, `ensure_inference_server_shut_down`)
  - `src/launch_sglang_server.rs`
- LLM request path currently assumes local sglang via `sglang_port`:
  - `src/llm_model/llm_model_traits.rs` (`LlmCliArgs`)
  - `src/llm_model/sglang_model_shared.rs` (POST to `http://localhost:{port}/generate`)
- Training launcher is currently local torchrun:
  - `src/launch_python_training.rs`
  - called from `src/orchestrator.rs::train_model`
- Model/checkpoint path rendering today is local filesystem templates:
  - `config/training/*.jinja`
  - helpers in `src/orchestrator.rs`

These are the integration points; most rollout and filtering code should stay untouched.

---

## Target Modal Interfaces

### 1) Inference (transparent sglang-compatible endpoint)

- Rust side should call an endpoint that behaves like sglang `/generate`.
- For Modal mode, requests go to a remote base URL instead of localhost port.
- Response shape should match existing parser expectations in `sglang_model_shared.rs`.

Minimum API:
- `POST /generate` (sglang-compatible)
- Optional health endpoint: `GET /healthz`

### 2) Training (remote job with progress streaming/polling)

- Rust sends:
  - model identity (`model_cli_name`, `config_nickname`, `epoch`)
  - training config payload (from current `PythonTrainingConfig` fields)
  - processed trajectories artifact
- Modal service:
  - pulls initial/base model from HF when needed
  - runs training remotely
  - stores trained model and checkpoints on Modal volume/object storage
  - exposes progress and final status

Minimum API:
- `POST /train/start` -> `{ job_id }`
- `GET /train/status/{job_id}` -> state + metrics/log cursor + artifact refs
- `POST /train/cancel/{job_id}`

### 3) Epoch-keyed training contract (no path resolution API)

Use logical model keys only:
- `model_cli_name`
- `config_nickname`
- `epoch`

Contract:
- Service reads model for `epoch` internally.
- Service writes trained output to internal location for `epoch + 1`.
- Client does not request or compose explicit model paths.

### 4) Server-side HTTP semantics (to implement)

`POST /train/start`
- Request headers:
  - `Idempotency-Key: {model_cli_name}/{config_nickname}/epoch_{epoch}`
- Success responses:
  - `200 OK` with `{ "job_id": "...", "created": true }` when new job is created.
  - `200 OK` with `{ "job_id": "...", "created": false }` when key already exists and same logical request is reused.
- Error responses:
  - `409 Conflict` with code `IDEMPOTENCY_KEY_REUSED_WITH_DIFFERENT_PAYLOAD` when key exists but payload hash differs.
  - `400 Bad Request` for malformed request.
  - `401/403` for auth failures.
  - `429` when rate-limited.
  - `500` for server/internal errors.

`PUT /train/upload_trajectory/{job_id}`
- Success responses:
  - `200 OK` when upload accepted.
  - `200 OK` when identical upload is replayed for same `job_id`.
- Error responses:
  - `404 Not Found` when `job_id` unknown.
  - `409 Conflict` when upload attempted after terminal state (`succeeded|failed|cancelled`).
  - `413 Payload Too Large` for oversized artifact.

`GET /train/status/{job_id}`
- Success response: `200 OK` with:
  - `status`: one of `queued|starting|running|succeeded|failed|cancelled`
  - `progress_message`: optional string
  - `progress_fraction`: optional float in `[0.0, 1.0]`
  - `error_code`: optional machine-readable string for terminal failures
  - `error_message`: optional human-readable message for terminal failures
- Error responses:
  - `404 Not Found` when `job_id` unknown.

`POST /train/cancel/{job_id}`
- Success responses:
  - `200 OK` with `cancelled=true` when transition applied.
  - `200 OK` with `cancelled=false` when already terminal (`succeeded|failed|cancelled`).
- Error responses:
  - `404 Not Found` when `job_id` unknown.

---

## Compatibility Strategy

Introduce explicit backend selection and keep HPC as default.

Proposed enum:
- `ComputeBackend::Hpc`
- `ComputeBackend::Modal`

Backend-specific behavior:
- Inference launch/shutdown:
  - `Hpc`: existing local sglang process behavior
  - `Modal`: no local process; keep a logical handle with remote endpoint metadata
- Training execution:
  - `Hpc`: existing `torchrun`
  - `Modal`: submit/poll remote job

Do not branch rollout or trajectory filtering logic by backend.

---

## Concrete Implementation Phases

## Phase 1 - Interface Refactor in Rust (no Modal runtime yet)

Objective: isolate backend-specific pieces behind traits/structs.

Work:
- Add compute backend config to orchestrator CLI (`src/bin/bin_orchestrator.rs`):
  - `--compute-backend` with default `hpc`
  - Modal settings inputs (base URL, auth token env key, timeout)
- Extend LLM endpoint args in `LlmCliArgs`:
  - keep `sglang_port: Option<u16>`
  - add `sglang_base_url: Option<String>`
- Update `SharedSglangLlmCallable` request URL construction:
  - if `sglang_base_url` set -> use `{base_url}/generate`
  - else use current localhost port behavior
- Introduce backend abstraction for orchestrator operations:
  - inference prepare/cleanup
  - training launch/wait

Acceptance:
- Existing HPC workflows unchanged.
- Unit/integration sanity check passes with only local backend.

## Phase 2 - Modal Inference Support

Objective: enable rollout inference against Modal-hosted sglang-compatible service.

Work:
- Add Modal inference client in Rust:
  - auth header injection
  - retry/backoff for transient errors
  - health check before rollout start
- In `launch_inference_server` path:
  - for Modal: skip local `uv run ... sglang.launch_server`
  - create handle with remote endpoint metadata
- Ensure logs clearly print backend and endpoint in use.

Acceptance:
- Validation and rollout collection run against Modal endpoint with no rollout code changes.
- Response parsing remains compatible.

## Phase 3 - Modal Training Support

Objective: move training compute to Modal jobs.

Work:
- Add Rust `modal_training_client` module:
  - submit job with serialized training config + trajectory artifact reference/upload
  - poll status and stream progress to existing TUI logger
  - return success/failure with error details
- In `train_model`:
  - branch on backend
  - keep current local config generation for reproducibility metadata
  - Modal path submits remote job instead of local torchrun
- Define artifact transfer contract:
  - Option A: upload sqlite trajectory file directly from Rust
  - Option B: pre-signed URL or object store handoff

Acceptance:
- End-to-end epoch train step succeeds on Modal.
- Failure and cancellation behavior is explicit and logged.

## Phase 4 - Epoch Resume + Idempotency

Objective: make resume robust without exposing physical model paths.

Work:
- Keep existing local path template logic for HPC; do not remove.
- Enforce stable logical key convention:
  - key = `{model_cli_name}/{config_nickname}/epoch_{epoch}`
- Make `/train/start` idempotent for logical epoch keys.
- Ensure service-side lookup handles resume/retry for completed and in-flight jobs.

Acceptance:
- Orchestrator can resume at any epoch by reusing logical epoch identifiers only.

## Phase 5 - Hardening and Operational Readiness

Work:
- Timeouts/retries/circuit-breaker style guards for all Modal API calls.
- Idempotency keys for training submission (`model/config/epoch`).
- Structured error mapping (HTTP/network/service errors).
- Add runbook docs for:
  - auth setup
  - endpoint config
  - job debugging
  - fallback to HPC mode

Acceptance:
- Documented recovery procedures.
- Clear fallback path to local/HPC backend.

---

## Proposed File-Level Change Plan

- `src/bin/bin_orchestrator.rs`
  - add backend/modal CLI flags
- `src/llm_model/llm_model_traits.rs`
  - extend `LlmCliArgs` for remote base URL
- `src/llm_model/sglang_model_shared.rs`
  - URL selection logic (base URL vs localhost port)
- `src/orchestrator.rs`
  - backend selection wiring
  - inference and training branching through backend abstraction
- `src/launch_sglang_server.rs`
  - keep for HPC path, minimal/no changes
- `src/launch_python_training.rs`
  - keep for HPC path, minimal/no changes
- New modules (suggested):
  - `src/compute_backend.rs`
  - `src/modal_client.rs`
  - `src/modal_training_client.rs`

---

## API/Data Contracts to Lock Early

Before implementation, freeze:
- Modal auth method (bearer token, key header, or signed request)
- Training trajectory upload protocol and max artifact size
- Training status schema (states, progress %, metric fields, log pagination)
- `/train/start` idempotency semantics and payload-hash policy
- Error schema (machine-readable code + human message)

Without these, Rust integration will churn.

---

## Test Plan

- Unit tests:
  - endpoint URL resolution logic
  - backend config parsing
  - model key generation
- Integration tests:
  - mock Modal API for inference `/generate`
  - mock training lifecycle (`start -> polling -> success/failure`)
- End-to-end smoke:
  - one short orchestration run on HPC backend (regression)
  - one short orchestration run on Modal backend
- Resume test:
  - restart from saved progress file mid-epoch in Modal mode

---

## Milestones and Deliverables

1. **M1: Backend Refactor PR**
   - No behavior change, HPC green
2. **M2: Modal Inference PR**
   - Rollouts via remote sglang-compatible endpoint
3. **M3: Modal Training PR**
   - Remote training jobs with progress reporting
4. **M4: Idempotent Resume PR**
   - Epoch-keyed start idempotency operational (`Idempotency-Key`)
5. **M5: Hardening + Docs PR**
   - production-readiness checks and runbooks

---

## Risks and Mitigations

- API mismatch with sglang response shape
  - Mitigation: strict compatibility tests using captured local responses
- Large trajectory artifact transfer cost/latency
  - Mitigation: compressed upload and/or object-store indirection
- Partial failures during long training jobs
  - Mitigation: idempotent submission, robust polling, resumable job tracking
- Drift between HPC and Modal behavior
  - Mitigation: backend-agnostic orchestration core + dual-backend smoke tests per release

---

## Reference

- Modal docs: https://modal.com/docs
