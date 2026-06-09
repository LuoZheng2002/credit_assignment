from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import modal
import modal.experimental

from src_py.experiment_identity import modal_function_name
from src_py.load_model_to_path import ensure_model_snapshot
from src_py.train.pathing import model_parent_dir, resolve_artifact_root_dir

MINUTES = 60
APP_NAME = "credit-assignment-sglang-service"
PORT = 8000
SGLANG_PORT = 30000
REGION = os.environ.get("MODAL_REGION", "us-east")
GPU = "H100:1"
MODEL_NAME = os.environ.get("SGLANG_MODEL_NAME", "Qwen/Qwen3.5-4B")
EXPERIMENT_KEYS_RAW = os.environ.get("MODAL_EXPERIMENT_KEYS", "")

HF_CACHE_PATH = "/root/.cache/huggingface"
JOBS_ROOT = Path("/mnt/service-state/jobs")
IDEMPOTENCY_ROOT = Path("/mnt/service-state/idempotency")
SERVICE_STATE_ROOT = Path("/mnt/service-state")

service_state_volume = modal.Volume.from_name(
    "credit-assignment-modal-service-state", create_if_missing=True
)
hf_cache_volume = modal.Volume.from_name("credit-assignment-hf-cache", create_if_missing=True)

base_image = (
    modal.Image.from_registry("lmsysorg/sglang:v0.5.10.post1-cu126")
    .entrypoint([])
    .pip_install("fastapi>=0.115.0", "uvicorn>=0.32.0", "requests>=2.32.0")
    .add_local_dir(".", remote_path="/root/credit_assignment")
    .env(
        {
            "HF_HUB_CACHE": HF_CACHE_PATH,
            "PYTHONPATH": "/root/credit_assignment",
        }
    )
)

app = modal.App(name=APP_NAME)

_MODAL_INFERENCE_LOCK = threading.Lock()
_MODAL_INFERENCE_PROCESS: subprocess.Popen[bytes] | None = None
_MODAL_INFERENCE_MODEL_NAME: str | None = None


def _configured_experiment_keys() -> list[str]:
    keys = [value.strip() for value in EXPERIMENT_KEYS_RAW.split(",") if value.strip()]
    unique: list[str] = []
    seen: set[str] = set()
    for key in keys:
        if key in seen:
            continue
        seen.add(key)
        unique.append(key)
    return unique


def _image_for_experiment(experiment_key: str) -> modal.Image:
    return base_image.env({"EXPERIMENT_KEY": experiment_key})


def _payload_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _key_filename(idempotency_key: str) -> str:
    key_hash = hashlib.sha256(idempotency_key.encode("utf-8")).hexdigest()
    return f"{key_hash}.json"


def _ensure_dirs() -> None:
    SERVICE_STATE_ROOT.mkdir(parents=True, exist_ok=True)
    JOBS_ROOT.mkdir(parents=True, exist_ok=True)
    IDEMPOTENCY_ROOT.mkdir(parents=True, exist_ok=True)


def _to_toml_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if value == float("inf") or value == float("-inf"):
            raise ValueError("inf is not supported in train_request.toml")
        if value != value:
            raise ValueError("nan is not supported in train_request.toml")
        return repr(value)
    if isinstance(value, str):
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    raise ValueError(f"Unsupported value type for train_request.toml: {type(value)}")


def _write_flat_toml(path: Path, payload: dict[str, Any]) -> None:
    lines: list[str] = []
    for key in sorted(payload.keys()):
        value = payload[key]
        if value is None:
            continue
        if isinstance(value, (dict, list)):
            raise ValueError(f"Nested training_config field is not supported for key: {key}")
        lines.append(f"{key} = {_to_toml_scalar(value)}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _wait_for_local_health(port: int, timeout_secs: int = 600) -> None:
    import requests

    deadline = time.time() + timeout_secs
    while time.time() < deadline:
        try:
            response = requests.get(f"http://127.0.0.1:{port}/health", timeout=2)
            if response.status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(1)
    raise TimeoutError(f"timed out waiting for local server health on port {port}")


def _ensure_modal_inference_sglang(model_name: str) -> None:
    global _MODAL_INFERENCE_PROCESS
    global _MODAL_INFERENCE_MODEL_NAME
    with _MODAL_INFERENCE_LOCK:
        if _MODAL_INFERENCE_PROCESS is not None and _MODAL_INFERENCE_PROCESS.poll() is None:
            if _MODAL_INFERENCE_MODEL_NAME == model_name:
                return
            _MODAL_INFERENCE_PROCESS.terminate()
            _MODAL_INFERENCE_PROCESS.wait(timeout=30)
        _MODAL_INFERENCE_PROCESS = subprocess.Popen(
            [
                "python",
                "-m",
                "sglang.launch_server",
                "--model-path",
                model_name,
                "--served-model-name",
                model_name,
                "--host",
                "127.0.0.1",
                "--port",
                str(SGLANG_PORT),
                "--tp",
                "1",
            ]
        )
        _MODAL_INFERENCE_MODEL_NAME = model_name
    _wait_for_local_health(SGLANG_PORT)


def _run_training_subprocess(
    training_config: dict[str, Any],
    trajectory_bytes: bytes,
    num_gpus: int,
    model_name: str,
) -> dict[str, Any]:
    if num_gpus != 1:
        return {
            "ok": False,
            "error_code": "MODAL_GPU_COUNT_UNSUPPORTED",
            "error": "modal_train currently supports num_gpus=1 only",
        }

    epoch = int(training_config.get("epoch", 0))
    if epoch == 0:
        artifact_root_dir = resolve_artifact_root_dir(training_config)
        model_cli_name = str(training_config["model_cli_name"])
        config_nickname = str(training_config["config_nickname"])
        parent_dir = model_parent_dir(artifact_root_dir, model_cli_name, config_nickname, epoch)
        if not (parent_dir / "model").exists():
            ensure_model_snapshot(parent_dir, model_name)

    with tempfile.TemporaryDirectory(prefix="modal_train_job_") as temp_dir:
        job_dir = Path(temp_dir)
        input_dir = job_dir / "input"
        input_dir.mkdir(parents=True, exist_ok=True)
        _write_flat_toml(job_dir / "train_request.toml", training_config)
        (input_dir / "training_trajectories.sqlite").write_bytes(trajectory_bytes)

        cmd = [
            "python",
            "-m",
            "src_py.train.main_from_config",
            "--job-folder-path",
            str(job_dir),
        ]
        termination_requested = threading.Event()
        child: subprocess.Popen[str] | None = None

        def _on_term(_: int, __: Any) -> None:
            termination_requested.set()
            nonlocal child
            if child is not None and child.poll() is None:
                child.terminate()

        previous_sigterm = signal.signal(signal.SIGTERM, _on_term)
        previous_sigint = signal.signal(signal.SIGINT, _on_term)
        try:
            child = subprocess.Popen(
                cmd,
                cwd="/root/credit_assignment",
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            while child.poll() is None:
                if termination_requested.is_set():
                    break
                time.sleep(0.5)
            _stdout_text, stderr_text = child.communicate(timeout=30)
            return_code = child.returncode
        finally:
            signal.signal(signal.SIGTERM, previous_sigterm)
            signal.signal(signal.SIGINT, previous_sigint)

        if termination_requested.is_set():
            return {
                "ok": False,
                "error_code": "CANCELLED_BY_SIGNAL",
                "error": "modal training subprocess received SIGTERM/SIGINT",
            }

        if return_code == 0:
            return {"ok": True, "message": "training completed"}
        return {
            "ok": False,
            "error_code": "TRAIN_PROCESS_FAILED",
            "error": (
                f"training subprocess failed rc={return_code}; stderr={stderr_text[-1000:]}"
            ),
        }


@app.function(
    image=base_image,
    gpu=GPU,
    region=REGION,
    min_containers=0,
    max_containers=1,
    timeout=20 * MINUTES,
    volumes={HF_CACHE_PATH: hf_cache_volume},
)
def modal_generate(payload: dict[str, Any], model_name: str = MODEL_NAME) -> dict[str, Any]:
    return _modal_generate_impl(payload, model_name)


def _modal_generate_impl(payload: dict[str, Any], model_name: str) -> dict[str, Any]:
    import requests

    _ensure_modal_inference_sglang(model_name)
    response = requests.post(
        f"http://127.0.0.1:{SGLANG_PORT}/generate",
        json=payload,
        timeout=600,
    )
    response.raise_for_status()
    return response.json()


@app.function(
    image=base_image,
    gpu=GPU,
    region=REGION,
    min_containers=0,
    max_containers=1,
    timeout=8 * 60 * MINUTES,
    volumes={HF_CACHE_PATH: hf_cache_volume, "/mnt/service-state": service_state_volume},
)
def modal_train(
    training_config: dict[str, Any],
    trajectory_bytes: bytes,
    num_gpus: int,
    model_name: str = MODEL_NAME,
) -> dict[str, Any]:
    return _modal_train_impl(training_config, trajectory_bytes, num_gpus, model_name)


def _modal_train_impl(
    training_config: dict[str, Any],
    trajectory_bytes: bytes,
    num_gpus: int,
    model_name: str,
) -> dict[str, Any]:
    return _run_training_subprocess(training_config, trajectory_bytes, num_gpus, model_name)


def _register_experiment_scoped_functions(experiment_key: str) -> None:
    generate_name = f"modal_generate__{experiment_key}"
    train_name = f"modal_train__{experiment_key}"
    experiment_image = _image_for_experiment(experiment_key)

    @app.function(
        name=generate_name,
        image=experiment_image,
        gpu=GPU,
        region=REGION,
        min_containers=0,
        max_containers=1,
        timeout=20 * MINUTES,
        volumes={HF_CACHE_PATH: hf_cache_volume},
    )
    def _modal_generate_experiment(payload: dict[str, Any], model_name: str) -> dict[str, Any]:
        return _modal_generate_impl(payload, model_name)

    @app.function(
        name=train_name,
        image=experiment_image,
        gpu=GPU,
        region=REGION,
        min_containers=0,
        max_containers=1,
        timeout=8 * 60 * MINUTES,
        volumes={HF_CACHE_PATH: hf_cache_volume, "/mnt/service-state": service_state_volume},
    )
    def _modal_train_experiment(
        training_config: dict[str, Any],
        trajectory_bytes: bytes,
        num_gpus: int,
        model_name: str,
    ) -> dict[str, Any]:
        expected = train_name
        actual = modal_function_name(
            "modal_train",
            str(training_config["model_cli_name"]),
            str(training_config["config_nickname"]),
        )
        if actual != expected:
            return {
                "ok": False,
                "error_code": "EXPERIMENT_KEY_MISMATCH",
                "error": f"expected {expected} but got {actual}",
            }
        return _modal_train_impl(training_config, trajectory_bytes, num_gpus, model_name)


for experiment_key in _configured_experiment_keys():
    _register_experiment_scoped_functions(experiment_key)


@app.cls(
    image=base_image,
    gpu=GPU,
    region=REGION,
    startup_timeout=20 * MINUTES,
    min_containers=0,
    max_containers=1,
    volumes={
        "/mnt/service-state": service_state_volume,
        HF_CACHE_PATH: hf_cache_volume,
    },
)
@modal.experimental.http_server(
    port=PORT,
    proxy_regions=[REGION],
    exit_grace_period=30,
)
@modal.concurrent(target_inputs=32)
class SglangTrainingService:
    @modal.enter()
    def startup(self) -> None:
        _ensure_dirs()
        self._lock = threading.Lock()
        self._job_threads: dict[str, threading.Thread] = {}
        self._job_processes: dict[str, subprocess.Popen[bytes]] = {}
        self._sglang_process = subprocess.Popen(
            [
                "python",
                "-m",
                "sglang.launch_server",
                "--model-path",
                MODEL_NAME,
                "--served-model-name",
                MODEL_NAME,
                "--host",
                "0.0.0.0",
                "--port",
                str(SGLANG_PORT),
                "--tp",
                "1",
                "--enable-metrics",
            ]
        )
        self._wait_for_sglang_ready()
        self._start_http_gateway()
        self._wait_for_gateway_ready()

    @modal.exit()
    def shutdown(self) -> None:
        gateway_server = getattr(self, "_gateway_server", None)
        if gateway_server is not None:
            gateway_server.should_exit = True
        gateway_thread = getattr(self, "_gateway_thread", None)
        if gateway_thread is not None and gateway_thread.is_alive():
            gateway_thread.join(timeout=10)
        if getattr(self, "_sglang_process", None) is not None:
            self._sglang_process.terminate()

    def _wait_for_sglang_ready(self, timeout_secs: int = 600) -> None:
        import requests

        deadline = time.time() + timeout_secs
        while time.time() < deadline:
            if self._sglang_process.poll() is not None:
                raise RuntimeError("sglang process exited before becoming ready")
            try:
                response = requests.get(f"http://127.0.0.1:{SGLANG_PORT}/health", timeout=2)
                if response.status_code == 200:
                    return
            except requests.RequestException:
                pass
            time.sleep(2)
        raise TimeoutError("Timed out waiting for sglang to become ready")

    def _start_http_gateway(self) -> None:
        from fastapi import FastAPI, Header, HTTPException, Request, Response
        from fastapi.responses import JSONResponse
        import requests
        import uvicorn

        service = self
        fastapi_app = FastAPI(title="Credit Assignment Modal SGLang Service")

        def _job_dir(job_id: str) -> Path:
            return JOBS_ROOT / job_id

        def _job_meta_path(job_id: str) -> Path:
            return _job_dir(job_id) / "job_meta.json"

        def _read_job_meta(job_id: str) -> dict[str, Any]:
            path = _job_meta_path(job_id)
            if not path.exists():
                raise HTTPException(status_code=404, detail=f"unknown job_id: {job_id}")
            return json.loads(path.read_text(encoding="utf-8"))

        def _write_job_meta(job_id: str, payload: dict[str, Any]) -> None:
            _job_meta_path(job_id).write_text(
                json.dumps(payload, ensure_ascii=True, sort_keys=True, indent=2),
                encoding="utf-8",
            )

        def _status_response(meta: dict[str, Any]) -> dict[str, Any]:
            response = {
                "status": meta.get("status", "queued"),
                "progress_message": meta.get("progress_message"),
                "progress_fraction": meta.get("progress_fraction"),
            }
            if meta.get("error_code"):
                response["error_code"] = meta["error_code"]
            if meta.get("error_message"):
                response["error_message"] = meta["error_message"]
            return response

        def _run_training_job(job_id: str) -> None:
            with service._lock:
                meta = _read_job_meta(job_id)
                if meta.get("status") in {"running", "succeeded", "failed", "cancelled"}:
                    return
                meta["status"] = "starting"
                meta["progress_message"] = "launching training subprocess"
                meta["progress_fraction"] = 0.05
                _write_job_meta(job_id, meta)

            job_dir = _job_dir(job_id)
            logs_dir = job_dir / "logs"
            logs_dir.mkdir(parents=True, exist_ok=True)
            train_log_path = logs_dir / "train.log"
            cmd = [
                "python",
                "-m",
                "src_py.train.main_from_config",
                "--job-folder-path",
                str(job_dir),
            ]

            with service._lock:
                meta = _read_job_meta(job_id)
                meta["status"] = "running"
                meta["progress_message"] = "training in progress"
                meta["progress_fraction"] = 0.2
                _write_job_meta(job_id, meta)

            with train_log_path.open("ab") as out:
                process = subprocess.Popen(
                    cmd,
                    cwd="/root/credit_assignment",
                    stdout=out,
                    stderr=out,
                    env=os.environ.copy(),
                )
                with service._lock:
                    service._job_processes[job_id] = process
                return_code = process.wait()
            with service._lock:
                service._job_processes.pop(job_id, None)

            with service._lock:
                meta = _read_job_meta(job_id)
                if meta.get("status") == "cancelled":
                    _write_job_meta(job_id, meta)
                    return
                if return_code == 0:
                    meta["status"] = "succeeded"
                    meta["progress_message"] = "training completed"
                    meta["progress_fraction"] = 1.0
                    meta["error_code"] = None
                    meta["error_message"] = None
                else:
                    meta["status"] = "failed"
                    meta["progress_message"] = "training failed"
                    meta["error_code"] = "TRAIN_PROCESS_FAILED"
                    meta["error_message"] = (
                        f"training process exited with code {return_code}; log={train_log_path}"
                    )
                    meta["progress_fraction"] = meta.get("progress_fraction") or 0.2
                _write_job_meta(job_id, meta)

        @fastapi_app.get("/health")
        def health() -> dict[str, str]:
            return {"status": "ok"}

        @fastapi_app.get("/healthz")
        def healthz() -> dict[str, str]:
            return {"status": "ok"}

        @fastapi_app.post("/generate")
        async def generate(request: Request) -> Response:
            payload = await request.json()
            upstream = requests.post(
                f"http://127.0.0.1:{SGLANG_PORT}/generate",
                json=payload,
                timeout=600,
            )
            return Response(
                content=upstream.content,
                status_code=upstream.status_code,
                media_type=upstream.headers.get("content-type", "application/json"),
            )

        @fastapi_app.post("/train/start")
        async def train_start(
            request: Request,
            idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
        ) -> dict[str, Any]:
            if idempotency_key is None or not idempotency_key.strip():
                raise HTTPException(status_code=400, detail="Idempotency-Key header is required")

            payload = await request.json()
            payload_hash = _payload_hash(payload)
            key_path = IDEMPOTENCY_ROOT / _key_filename(idempotency_key.strip())
            with service._lock:
                if key_path.exists():
                    existing = json.loads(key_path.read_text(encoding="utf-8"))
                    if existing.get("payload_hash") != payload_hash:
                        return JSONResponse(
                            status_code=409,
                            content={
                                "error_code": "IDEMPOTENCY_KEY_REUSED_WITH_DIFFERENT_PAYLOAD",
                                "error_message": "payload hash mismatch for idempotency key",
                            },
                        )
                    return {"job_id": existing["job_id"], "created": False}

                job_id = str(uuid.uuid4())
                job_dir = _job_dir(job_id)
                (job_dir / "input").mkdir(parents=True, exist_ok=True)
                (job_dir / "output").mkdir(parents=True, exist_ok=True)

                training_config = payload.get("training_config")
                if not isinstance(training_config, dict):
                    raise HTTPException(
                        status_code=400,
                        detail="training_config object is required",
                    )
                _write_flat_toml(job_dir / "train_request.toml", training_config)

                meta = {
                    "job_id": job_id,
                    "status": "queued",
                    "progress_message": "waiting for trajectory upload",
                    "progress_fraction": 0.0,
                    "error_code": None,
                    "error_message": None,
                    "created_at": time.time(),
                    "uploaded": False,
                }
                _write_job_meta(job_id, meta)
                key_path.write_text(
                    json.dumps(
                        {
                            "idempotency_key": idempotency_key.strip(),
                            "payload_hash": payload_hash,
                            "job_id": job_id,
                        },
                        ensure_ascii=True,
                        sort_keys=True,
                    ),
                    encoding="utf-8",
                )
                return {"job_id": job_id, "created": True}

        @fastapi_app.put("/train/upload_trajectory/{job_id}")
        async def upload_trajectory(job_id: str, request: Request) -> dict[str, Any]:
            content = await request.body()
            if not content:
                raise HTTPException(status_code=400, detail="trajectory upload body is empty")
            with service._lock:
                meta = _read_job_meta(job_id)
                if meta.get("status") in {"succeeded", "failed", "cancelled"}:
                    raise HTTPException(
                        status_code=409,
                        detail=f"job {job_id} already terminal: {meta.get('status')}",
                    )

                trajectory_path = _job_dir(job_id) / "input" / "training_trajectories.sqlite"
                if trajectory_path.exists():
                    existing = trajectory_path.read_bytes()
                    if hashlib.sha256(existing).hexdigest() == hashlib.sha256(content).hexdigest():
                        return {"accepted": True, "replayed": True}
                trajectory_path.write_bytes(content)

                meta["uploaded"] = True
                meta["status"] = "starting"
                meta["progress_message"] = "trajectory uploaded; scheduling training"
                meta["progress_fraction"] = 0.1
                _write_job_meta(job_id, meta)

                thread = service._job_threads.get(job_id)
                if thread is None or not thread.is_alive():
                    thread = threading.Thread(target=_run_training_job, args=(job_id,), daemon=True)
                    service._job_threads[job_id] = thread
                    thread.start()
            return {"accepted": True, "replayed": False}

        @fastapi_app.get("/train/status/{job_id}")
        def train_status(job_id: str) -> dict[str, Any]:
            with service._lock:
                meta = _read_job_meta(job_id)
                return _status_response(meta)

        @fastapi_app.post("/train/cancel/{job_id}")
        def train_cancel(job_id: str) -> dict[str, bool]:
            with service._lock:
                meta = _read_job_meta(job_id)
                if meta.get("status") in {"succeeded", "failed", "cancelled"}:
                    return {"cancelled": False}
                process = service._job_processes.get(job_id)
                if process is not None and process.poll() is None:
                    process.terminate()
                meta["status"] = "cancelled"
                meta["progress_message"] = "cancelled by user"
                meta["error_code"] = "CANCELLED_BY_USER"
                meta["error_message"] = "job cancelled by user request"
                _write_job_meta(job_id, meta)
                return {"cancelled": True}

        @fastapi_app.post("/admin/shutdown")
        def admin_shutdown() -> dict[str, bool]:
            def _delayed_exit() -> None:
                time.sleep(0.5)
                os._exit(0)

            threading.Thread(target=_delayed_exit, daemon=True).start()
            return {"shutting_down": True}

        config = uvicorn.Config(
            fastapi_app,
            host="0.0.0.0",
            port=PORT,
            log_level="info",
            timeout_keep_alive=60,
        )
        server = uvicorn.Server(config)
        self._gateway_server = server
        self._gateway_thread = threading.Thread(target=server.run, daemon=True)
        self._gateway_thread.start()

    def _wait_for_gateway_ready(self, timeout_secs: int = 60) -> None:
        import requests

        deadline = time.time() + timeout_secs
        while time.time() < deadline:
            try:
                response = requests.get(f"http://127.0.0.1:{PORT}/health", timeout=2)
                if response.status_code == 200:
                    return
            except requests.RequestException:
                pass
            time.sleep(1)
        raise TimeoutError("Timed out waiting for gateway server to become ready")


@app.local_entrypoint()
def show_url() -> None:
    print("Deploy with: modal deploy modal_app.py")
