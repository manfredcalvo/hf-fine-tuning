# Databricks notebook source
# MAGIC %md
# MAGIC # Log fine-tuned model and deploy to a Databricks Model Serving endpoint
# MAGIC
# MAGIC 1. (Optional) Smoke-test the model locally with a vLLM OpenAI-compatible server.
# MAGIC 2. Log the model to MLflow with a custom vLLM entrypoint (`task: llm/v1/chat`).
# MAGIC 3. Register it to Unity Catalog with `env_pack="databricks_model_serving"`.
# MAGIC 4. Create (or update) a GPU Model Serving endpoint and query it.

# COMMAND ----------

# MAGIC %md
# MAGIC Serving requirements. For serverless GPU jobs, dependencies must be installed in the
# MAGIC notebook (the Environments panel is not supported for serverless GPU scheduled jobs).
# MAGIC
# MAGIC The Model Serving GPU containers run an NVIDIA driver at CUDA 12.4, so vLLM's default
# MAGIC PyPI wheel (built for CUDA 13) crashes there with "NVIDIA driver too old". Install the
# MAGIC `+cu129` variant wheel and CUDA 12.9 PyTorch instead — CUDA 12.x runtimes work on the
# MAGIC 12.4 driver via minor-version compatibility, and `env_pack` snapshots this exact
# MAGIC environment into the serving container.

# COMMAND ----------

# pip 24.1+ treats conflicts between base-image packages and ephemeral-env packages as
# errors (exit 1) instead of warnings (exit 0). Downgrade pip to <24.1 first so the
# conflict with databricks-serverless-gpu's mlflow<3.0 requirement is only a warning.
# MAGIC %pip install "pip<24.1"
# MAGIC %pip install --upgrade https://github.com/vllm-project/vllm/releases/download/v0.23.0/vllm-0.23.0+cu129-cp38-abi3-manylinux_2_28_x86_64.whl "transformers>=5.5.0" openai==2.17.0 mlflow==3.12.0 hf_transfer==0.1.9 databricks-sdk "fastapi<0.137" --extra-index-url https://download.pytorch.org/whl/cu129

# COMMAND ----------

# MAGIC %md
# MAGIC Slim the environment before it gets snapshotted by `env_pack`. FlashInfer is optional
# MAGIC for vLLM (its sampler is disabled in our entrypoint and it can't JIT on the serving
# MAGIC image). Ray is only needed for multi-node executors. Uninstalling BEFORE the smoke test
# MAGIC means we validate the exact slimmed environment that serving will run.

# COMMAND ----------

# MAGIC %pip uninstall -y flashinfer-python flashinfer-cubin flashinfer-jit-cache ray
# MAGIC %restart_python

# COMMAND ----------

import site, subprocess

sp = site.getsitepackages()[0]

def sh(cmd):
    return subprocess.run(["bash", "-lc", cmd], capture_output=True, text=True).stdout

print("site-packages:", sp)
print("--- top 25 packages by size (MB) ---")
print(sh(f"du -sm {sp}/* 2>/dev/null | sort -rn | head -25"))

sh(f"find {sp} -name '__pycache__' -type d -prune -exec rm -rf {{}} + 2>/dev/null")
sh(f"find {sp}/nvidia -name '*_static*' -delete 2>/dev/null; find {sp} -name '*.a' -delete 2>/dev/null")
sh("pip cache purge >/dev/null 2>&1 || true; rm -rf ~/.cache/pip")

print("--- total after trim ---")
print(sh(f"du -sh {sp}"))

# COMMAND ----------

# MAGIC %sh
# MAGIC nvidia-smi

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

dbutils.widgets.text("CATALOG", "meli_demo")
dbutils.widgets.text("SCHEMA", "default")
dbutils.widgets.text("MODEL_NAME", "deepseek_rag_chat_model")
dbutils.widgets.text("MODEL_PATH", "/Volumes/lucas_catalog/default/models/deep_seek_ft_model_1/")
dbutils.widgets.text("BASE_MODEL", "deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
dbutils.widgets.text("ENDPOINT_NAME", "deepseek_ft_playground_rag")
dbutils.widgets.text("train_data_table_name_ft", "chat_completion_training_dataset")
dbutils.widgets.text("SMOKE_TEST", "false")

CATALOG = dbutils.widgets.get("CATALOG")
SCHEMA = dbutils.widgets.get("SCHEMA")
MODEL_NAME = dbutils.widgets.get("MODEL_NAME")
MODEL_PATH = dbutils.widgets.get("MODEL_PATH")
BASE_MODEL = dbutils.widgets.get("BASE_MODEL")
ENDPOINT_NAME = dbutils.widgets.get("ENDPOINT_NAME")
train_data_table_name_ft = dbutils.widgets.get("train_data_table_name_ft")
SMOKE_TEST = dbutils.widgets.get("SMOKE_TEST").strip().lower() == "true"

UC_MODEL_NAME = f"{CATALOG}.{SCHEMA}.{MODEL_NAME}"
SERVED_MODEL_NAME = MODEL_NAME

# Local path used by both the smoke test and the serving container.
LOCAL_MODEL_PATH = "/tmp/model_weights"

# vLLM tuning for DeepSeek LLaMA 8B (float16, ~16 GiB)
DTYPE = "float16"
MAX_MODEL_LEN = 8192
GPU_MEMORY_UTILIZATION = 0.85
TENSOR_PARALLEL = 1

# Allowlisted ports for Serverless GPU notebooks: 3000-3999. Model Serving requires 8080.
LOCAL_PORT = 3080
SERVING_PORT = 8080

from databricks.sdk.service.serving import ServingModelWorkloadType
WORKLOAD_TYPE = ServingModelWorkloadType.GPU_MEDIUM
PROVISIONED_CONCURRENCY = 4
SCALE_TO_ZERO_ENABLED = False

import os, tempfile

# Work from a temp directory on local disk so large model files don't go to the workspace.
workdir = tempfile.mkdtemp()
os.chdir(workdir)

# Speed up HuggingFace downloads.
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ.pop("OPENSSL_FORCE_FIPS_MODE", None)

print(f"Working directory : {workdir}")
print(f"Base model        : {BASE_MODEL}")
print(f"Deploying         : {UC_MODEL_NAME} -> endpoint '{ENDPOINT_NAME}'")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Define the vLLM entrypoint
# MAGIC
# MAGIC Single entrypoint used for both the smoke test and the serving container.
# MAGIC Model weights are too large to embed as MLflow artifacts (~16 GiB), so the serving
# MAGIC container downloads them from HuggingFace at startup via `hf download`. The smoke test
# MAGIC runs the exact same command, which also downloads before starting vLLM.

# COMMAND ----------

def entrypoint(port: int) -> str:
    # VLLM_USE_FLASHINFER_SAMPLER / VLLM_WORKER_MULTIPROC_METHOD are shell-level assignments
    # so they apply only to the python subprocess and don't affect hf download.
    vllm_cmd = " ".join([
        "VLLM_USE_FLASHINFER_SAMPLER=0",
        "VLLM_WORKER_MULTIPROC_METHOD=fork",
        "python", "-u", "-m", "vllm.entrypoints.openai.api_server",
        "--model", LOCAL_MODEL_PATH,
        "--served-model-name", SERVED_MODEL_NAME,
        "--host", "0.0.0.0",
        "--port", str(port),
        "--dtype", DTYPE,
        "--max-model-len", str(MAX_MODEL_LEN),
        "--gpu-memory-utilization", str(GPU_MEMORY_UTILIZATION),
        "--tensor-parallel-size", str(TENSOR_PARALLEL),
        "--trust-remote-code",
    ])
    # Pre-create the metadata dir to avoid a PermissionError race in hf download's
    # parallel threads on the overlay filesystem.
    download_cmd = (
        f"mkdir -p {LOCAL_MODEL_PATH}/.cache/huggingface/download"
        f" && hf download {BASE_MODEL} --local-dir {LOCAL_MODEL_PATH}"
    )
    # Wrap the entire shell session — including hf download — so that OPENSSL_FORCE_FIPS_MODE
    # is absent and OPENSSL_CONF=/dev/null is set before any OpenSSL code runs.
    # Databricks images set OPENSSL_FORCE_FIPS_MODE=0, which the RHEL OpenSSL bundled in
    # opencv>=4.13 (a vLLM dependency) treats as "enable FIPS", aborting with
    # FATAL FIPS SELFTEST FAILURE. OPENSSL_CONF=/dev/null prevents loading
    # /etc/ssl/openssl.cnf which activates the FIPS provider on FIPS-enabled kernels.
    return (
        "env -u OPENSSL_FORCE_FIPS_MODE"
        " CRYPTOGRAPHY_OPENSSL_NO_LEGACY=1"
        " OPENSSL_CONF=/dev/null"
        f" bash -c '{download_cmd} && {vllm_cmd}'"
    )

print(entrypoint(SERVING_PORT))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verify HuggingFace model accessibility

# COMMAND ----------

import os
from huggingface_hub import model_info as hf_model_info

print(f"Checking HuggingFace model: {BASE_MODEL}")
try:
    info = hf_model_info(BASE_MODEL)
    print(f"  Model ID : {info.modelId}")
    print(f"  Private  : {info.private}")
    print(f"  Pipeline : {info.pipeline_tag}")
except Exception as e:
    raise RuntimeError(f"Cannot access HuggingFace model '{BASE_MODEL}': {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Smoke-test the model in the notebook (optional)
# MAGIC
# MAGIC Starts vLLM on the notebook GPU using the local copy, waits until ready,
# MAGIC sends one chat request, then stops it. Set SMOKE_TEST=true to enable.

# COMMAND ----------

import subprocess, sys, time, requests

# Pre-flight: verify vLLM is importable and GPU is visible before launching subprocess.
preflight = subprocess.run(
    [sys.executable, "-c",
     "import vllm; import torch; "
     "print('vLLM:', vllm.__version__); "
     "print('CUDA available:', torch.cuda.is_available()); "
     "print('GPU count:', torch.cuda.device_count()); "
     "[print(f'GPU {i}:', torch.cuda.get_device_name(i), "
     "f'{torch.cuda.get_device_properties(i).total_memory/1e9:.1f} GB') "
     "for i in range(torch.cuda.device_count())]"],
    capture_output=True, text=True, env={**os.environ}
)
print("=== Pre-flight ===")
print(preflight.stdout)
if preflight.returncode != 0:
    print("STDERR:", preflight.stderr)
    raise RuntimeError("vLLM import pre-flight failed — see stderr above.")

if SMOKE_TEST:
    # Download model in-process: the notebook kernel has the HF credentials that
    # subprocess shells may not inherit (Databricks secrets injected at kernel level).
    from huggingface_hub import snapshot_download as _snap
    os.makedirs(f"{LOCAL_MODEL_PATH}/.cache/huggingface/download", exist_ok=True)
    _snap(BASE_MODEL, local_dir=LOCAL_MODEL_PATH)

    # FIPS constraints on Databricks serverless GPU notebooks:
    # 1. ssl.SSLError: PYTHONPATH includes /databricks/python where requests 2.32+ pre-creates
    #    SSLContext at import time. The kernel FIPS flag (/proc/sys/crypto/fips_enabled=1) is
    #    checked at the C level inside OpenSSL — OPENSSL_CONF=/dev/null is insufficient because
    #    the kernel check happens before any config file is read. Fix: patch
    #    urllib3.create_urllib3_context to return a no-op FakeSSLContext on failure so vLLM
    #    imports cleanly. All smoke-test traffic is plain HTTP so no real SSL is needed.
    # 2. FATAL FIPS SELFTEST FAILURE (SIGABRT, exit -6): triggered by C extensions with bundled
    #    RHEL OpenSSL when OPENSSL_FORCE_FIPS_MODE=0 is in env (opencv ≥ 4.13) or when
    #    HF_HUB_ENABLE_HF_TRANSFER=1 loads hf_transfer's Rust/PyO3 OpenSSL. Fix: strip both
    #    from the subprocess env. Also add OPENSSL_CONF=/dev/null as belt-and-suspenders.
    # Using sys.executable (no bash) avoids bash -lc sourcing system profiles that re-inject
    # OPENSSL_FORCE_FIPS_MODE=0 — the variable was already popped in the config cell above.
    with open("/tmp/vllm_wrapper.py", "w") as _wf:
        _wf.write(
            "import sys, faulthandler\n"
            "faulthandler.enable()  # print Python traceback on SIGABRT/SIGSEGV\n"
            "print('[W1] urllib3 patch', flush=True)\n"
            "try:\n"
            "    import urllib3.util.ssl_ as _u3ssl\n"
            "    _orig_ctx = _u3ssl.create_urllib3_context\n"
            "    class _FakeSSLContext:\n"
            "        def __getattr__(self, name): return lambda *a, **kw: None\n"
            "    def _patched_ctx(*a, **kw):\n"
            "        try: return _orig_ctx(*a, **kw)\n"
            "        except Exception: return _FakeSSLContext()\n"
            "    _u3ssl.create_urllib3_context = _patched_ctx\n"
            "    print('[W1] urllib3 patch OK', flush=True)\n"
            "except Exception as e: print('[W1] urllib3 patch FAILED:', e, flush=True)\n"
            "print('[W2] torch.distributed gloo patch', flush=True)\n"
            "try:\n"
            "    import torch.distributed as _dist\n"
            "    _orig_init = _dist.init_process_group\n"
            "    def _patched_init(*a, **kw):\n"
            "        kw['backend'] = 'gloo'\n"
            "        return _orig_init(*a, **kw)\n"
            "    _dist.init_process_group = _patched_init\n"
            "    print('[W2] gloo patch OK', flush=True)\n"
            "except Exception as e: print('[W2] gloo patch FAILED:', e, flush=True)\n"
            "import ctypes, ctypes.util\n"
            "for _lib in ['ssl', 'crypto', 'nccl', 'cudnn', 'cublas']:\n"
            "    print(f'[D] {_lib}: {ctypes.util.find_library(_lib)}', flush=True)\n"
            "print('[W3] import torch', flush=True)\n"
            "import torch\n"
            "print('[W3] torch imported, cuda:', torch.cuda.is_available(), flush=True)\n"
            "print('[W4] torch.cuda device count:', torch.cuda.device_count(), flush=True)\n"
            "print('[W5] creating CUDA tensor (first real GPU op)', flush=True)\n"
            "t = torch.zeros(4).cuda()\n"
            "print('[W5] CUDA tensor OK:', t.device, flush=True)\n"
            "del t; torch.cuda.synchronize()\n"
            "print('[W6] import vllm top-level only', flush=True)\n"
            "import vllm\n"
            "print('[W6] vllm imported OK, version:', vllm.__version__, flush=True)\n"
            "print('[W7] calling runpy.run_module vllm', flush=True)\n"
            "import runpy\n"
            "runpy.run_module('vllm.entrypoints.openai.api_server', run_name='__main__', alter_sys=True)\n"
        )

    _STRIP = {"OPENSSL_FORCE_FIPS_MODE", "HF_HUB_ENABLE_HF_TRANSFER"}
    _env_vllm = {k: v for k, v in os.environ.items() if k not in _STRIP}
    _env_vllm.update({
        "CRYPTOGRAPHY_OPENSSL_NO_LEGACY": "1",
        "OPENSSL_CONF": "/dev/null",
        "VLLM_USE_FLASHINFER_SAMPLER": "0",
        "VLLM_WORKER_MULTIPROC_METHOD": "fork",
    })

    log = open("process.log", "w")
    proc = subprocess.Popen(
        [sys.executable, "-u", "/tmp/vllm_wrapper.py",
         "--model", LOCAL_MODEL_PATH,
         "--served-model-name", SERVED_MODEL_NAME,
         "--host", "0.0.0.0",
         "--port", str(LOCAL_PORT),
         "--dtype", DTYPE,
         "--max-model-len", str(MAX_MODEL_LEN),
         "--gpu-memory-utilization", str(GPU_MEMORY_UTILIZATION),
         "--tensor-parallel-size", str(TENSOR_PARALLEL),
         "--trust-remote-code",
         "--disable-custom-all-reduce"],
        stdout=log,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
        env=_env_vllm,
    )

    # Wait for vLLM to come up, streaming its startup log into the cell output so
    # progress (engine init, checkpoint shard loading, CUDA graphs) is visible live.
    # Fails fast if the process crashes instead of waiting out the full deadline.
    deadline = time.time() + 45 * 60
    ready = False
    with open("process.log") as lf:
        while time.time() < deadline:
            for line in lf.readlines():
                print(line, end="", flush=True)
            try:
                if requests.get(f"http://localhost:{LOCAL_PORT}/health", timeout=5).status_code == 200:
                    ready = True
                    break
            except requests.exceptions.RequestException:
                pass
            if proc.poll() is not None:
                _remaining = lf.read()
                print(_remaining, flush=True)
                with open("process.log") as _lf:
                    _tail = "".join(_lf.readlines()[-150:])
                raise RuntimeError(
                    f"vLLM exited with code {proc.returncode} before becoming ready.\n"
                    f"=== Last 150 lines of process.log ===\n{_tail}\n=== END ==="
                )
            time.sleep(5)
        print(lf.read(), flush=True)

    if not ready:
        raise RuntimeError("vLLM did not become ready within 45 minutes (see log above).")

    resp = requests.post(
        f"http://localhost:{LOCAL_PORT}/invocations",
        json={"messages": [{"role": "user", "content": "What is Databricks? Reply in one sentence."}]},
    )
    resp.raise_for_status()
    print(resp.json()["choices"][0]["message"]["content"])
else:
    print("SMOKE_TEST=false, skipping local vLLM test.")

# COMMAND ----------

if SMOKE_TEST:
    subprocess.run(["pkill", "-f", "vllm.entrypoints.openai.api_server"])
    time.sleep(10)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Log the model with the custom entrypoint

# COMMAND ----------

import mlflow
from mlflow.pyfunc.model import ChatModel, ChatCompletionResponse

# Required placeholder — Model Serving runs the entrypoint, not python_model.predict.
class LLMModel(ChatModel):
    def predict(self, context, messages, params):
        return ChatCompletionResponse.from_dict({"choices": []})

dataset_info = mlflow.data.from_spark(
    spark.table(f"{CATALOG}.{SCHEMA}.{train_data_table_name_ft}"),
    table_name=f"{CATALOG}.{SCHEMA}.{train_data_table_name_ft}",
    version="1.0"
)

with mlflow.start_run():
    mlflow.log_input(dataset_info, context="training")

    model_info = mlflow.pyfunc.log_model(
        name=SERVED_MODEL_NAME,
        python_model=LLMModel(),
        artifacts=None,
        metadata={
            "task": "llm/v1/chat",
            "entrypoint": entrypoint(SERVING_PORT),
        },
        extra_pip_requirements=[
            "mlflow==3.12.0",
        ],
    )

print(f"Logged model: {model_info.model_uri}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Register the model to Unity Catalog

# COMMAND ----------

# env_pack is required — custom LLM Serving depends on Serverless Optimized Deployments.
model_version = mlflow.register_model(
    model_info.model_uri, UC_MODEL_NAME, env_pack="databricks_model_serving"
)
print(f"Registered {UC_MODEL_NAME} version {model_version.version}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create (or update) the serving endpoint

# COMMAND ----------

import time
from datetime import timedelta
from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import NotFound, ResourceConflict
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput

served_entity = ServedEntityInput(
    entity_name=UC_MODEL_NAME,
    entity_version=str(model_version.version),
    workload_type=WORKLOAD_TYPE,
    min_provisioned_concurrency=PROVISIONED_CONCURRENCY,
    max_provisioned_concurrency=PROVISIONED_CONCURRENCY,
    scale_to_zero_enabled=SCALE_TO_ZERO_ENABLED,
)

w = WorkspaceClient()

# Pre-flight: if the endpoint exists and is already being updated, wait for the
# in-progress update to finish before issuing a new one. GPU endpoint deployments
# can take well over an hour, and a timed-out _and_wait call leaves the endpoint
# in IN_PROGRESS — a second attempt would hit ResourceConflict immediately.
def _wait_for_idle(name, deadline_s=90 * 60):
    try:
        ep = w.serving_endpoints.get(name)
    except NotFound:
        return
    while ep.state and str(ep.state.config_update) == "EndpointStateConfigUpdate.IN_PROGRESS":
        if time.time() > deadline_s:
            raise TimeoutError(f"Endpoint {name} still updating after 90 minutes")
        print(f"Endpoint {name} is updating — waiting 60s before retry...")
        time.sleep(60)
        ep = w.serving_endpoints.get(name)

_wait_for_idle(ENDPOINT_NAME)

try:
    w.serving_endpoints.get(ENDPOINT_NAME)
    print(f"Endpoint '{ENDPOINT_NAME}' exists, updating to version {model_version.version}...")
    w.serving_endpoints.update_config_and_wait(
        name=ENDPOINT_NAME,
        served_entities=[served_entity],
        timeout=timedelta(minutes=120),
    )
except NotFound:
    print(f"Creating endpoint '{ENDPOINT_NAME}'...")
    w.serving_endpoints.create_and_wait(
        name=ENDPOINT_NAME,
        config=EndpointCoreConfigInput(name=ENDPOINT_NAME, served_entities=[served_entity]),
        timeout=timedelta(minutes=120),
    )

# Final safety check
ep = w.serving_endpoints.get(ENDPOINT_NAME)
if str(ep.state.ready) == "ENDPOINT_STATE_FAILED":
    raise RuntimeError(f"Serving endpoint {ENDPOINT_NAME} deployment failed: {ep.state.message}")

print(f"Endpoint '{ENDPOINT_NAME}' is ready.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Query the ready endpoint

# COMMAND ----------

from databricks.sdk.service.serving import ChatMessage, ChatMessageRole

resp = w.serving_endpoints.query(
    name=ENDPOINT_NAME,
    messages=[ChatMessage(role=ChatMessageRole.USER, content="What is Databricks? Reply in one sentence.")],
)
print(resp.choices[0].message.content)

# COMMAND ----------

dbutils.jobs.taskValues.set("model_version", model_version.version)
