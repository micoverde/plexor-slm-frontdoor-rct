# Plexor SLM Front-Door Experiment Design

**Version 2.0 — March 2026**
**Status:** Ready for Review
**Classification:** Confidential — Plexor Labs

---

## 1. Purpose

This document specifies the end-to-end experiment design for validating the Plexor SLM front-door routing hypothesis: that a cheap small language model (≤4B parameters) inserted at the pipeline entry point can classify tasks, compress prompts, and quality-gate responses well enough to reduce downstream inference cost by ≥25% while maintaining or improving output quality — all within a P95 latency budget of ≤200ms.

The experiment runs in four progressive stages, from offline model benchmarking through live randomised controlled trial (RCT), with all data tracked in a local MLflow instance backed by PostgreSQL and Azure Blob Storage.

---

## 2. Model Candidates

### 2.1 Why Run Both Phi and Qwen

The v1 locked pairwise benchmark (60-case corpus, RunPod A40) demonstrated that neither model family dominates across all six task families:

| Family | Phi-3.5-mini | Qwen2.5-1.5B | Winner |
|---|---|---|---|
| code_complex | 1.0 | 0.2 | Phi |
| code_simple | 0.0 | 0.0 | Neither |
| cot_complex | 1.0 | 0.0 | Phi |
| cot_simple | 0.8 | 0.0 | Phi |
| hybrid_agentic | 0.3 | 1.0 | Qwen |
| hybrid_generative | 0.0 | 0.0 | Neither |

This split demands that we test both families — plus newer model versions that may close the weak-family gaps. Additionally, the latency gap is massive (Phi median 12,133ms vs Qwen median 1,542ms on A40), so any production policy must weigh quality against speed.

### 2.2 Candidate Matrix

| Model ID | Params | Quantisation | VRAM (est.) | Role in Experiment |
|---|---|---|---|---|
| **Phi-4-mini-instruct** | 3.8B | 4-bit NF4 | ~3.5 GB | Primary Phi candidate (spec target) |
| **Phi-3.5-mini-instruct** | 3.8B | 4-bit NF4 | ~3.5 GB | Baseline continuity from v1 benchmark |
| **Qwen2.5-3B-Instruct** | 3.0B | 4-bit NF4 | ~2.8 GB | Primary Qwen candidate (larger than v1 1.5B) |
| **Qwen2.5-1.5B-Instruct** | 1.5B | 4-bit NF4 | ~1.5 GB | Baseline continuity from v1 benchmark |

All four models fit within the 16 GB VRAM ceiling of a single T4 GPU. Models are loaded one at a time during benchmark runs; never concurrently.

### 2.3 Decision: Run All Four, Crown Two

Run the Stage 1 offline benchmark across all four models on the same corpus, same hardware, same decoding contract. The top-performing Phi and top-performing Qwen advance to Stage 2. The final production policy may be a single model or a hybrid split (e.g., Phi-4-mini for code/CoT families, Qwen2.5-3B for hybrid_agentic).

---

## 3. Azure Infrastructure

### 3.1 GPU VM Recommendation

**Primary: Standard_NC8as_T4_v3**

| Spec | Value |
|---|---|
| GPU | 1× NVIDIA T4 (16 GB VRAM, Turing arch) |
| vCPUs | 8× AMD EPYC 7V12 (Rome) |
| RAM | 56 GB |
| Temp Storage | 360 GB SSD |
| On-Demand Price | ~$0.752/hr (East US, Linux) |
| Spot Price | ~$0.35/hr (variable, East US) |
| Region | East US (matches Foundry endpoints) |

**Why this VM:**

- The T4's 16 GB VRAM comfortably fits any single 4-bit quantized ≤4B model with room for KV cache. No need for A100-class hardware for SLM inference.
- 56 GB system RAM handles MLflow server, data loading, sentence-transformer embeddings for compression validation, and benchmark orchestration concurrently.
- 8 vCPUs are sufficient for the rule-based compressor (Stage 1), regex hypervisor (Stage 1), and data pipeline work.
- At ~$0.75/hr, a full 40-hour experiment week costs ~$30. Spot pricing cuts this to ~$14.

**Alternative (budget-constrained): Standard_NC4as_T4_v3**

| Spec | Value |
|---|---|
| GPU | 1× NVIDIA T4 (16 GB VRAM) |
| vCPUs | 4× AMD EPYC 7V12 |
| RAM | 28 GB |
| On-Demand Price | ~$0.526/hr |

This works for pure inference benchmarking but will be tight when running MLflow + model inference + embedding validation concurrently. Use the NC8 if budget allows.

### 3.2 VM Setup Script

```bash
# Provision via Azure CLI
az vm create \
  --resource-group plexor-rg \
  --name plexor-slm-bench \
  --image Canonical:ubuntu-24_04-lts:server:latest \
  --size Standard_NC8as_T4_v3 \
  --priority Spot \
  --eviction-policy Deallocate \
  --max-price 0.40 \
  --admin-username plexor \
  --generate-ssh-keys \
  --os-disk-size-gb 256 \
  --location eastus

# Attach data disk for MLflow artifacts
az vm disk attach \
  --resource-group plexor-rg \
  --vm-name plexor-slm-bench \
  --name plexor-slm-data \
  --size-gb 128 \
  --new
```

### 3.3 GPU Driver & Runtime Stack

```bash
# NVIDIA driver + CUDA toolkit
sudo apt-get update && sudo apt-get install -y nvidia-driver-545 nvidia-cuda-toolkit

# Verify GPU
nvidia-smi  # should show T4, 16 GB VRAM

# Python environment
conda create -n plexor python=3.11 -y
conda activate plexor

# ── Runtime 1: transformers + bitsandbytes (Stage 1–2 offline benchmark) ──
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install transformers accelerate bitsandbytes
pip install flash-attn --no-build-isolation  # optional; T4 supports flash-attn v2
pip install sentence-transformers             # compression similarity validation

# ── Runtime 2: vLLM (Stage 3–4 production-path serving) ──
pip install vllm  # installs its own torch if needed; verify no version conflict

# Verify vLLM can see GPU
python -c "from vllm import LLM; print('vLLM OK')"

# ── MLflow + observability ──
pip install mlflow psycopg2-binary boto3
pip install prometheus-client structlog fastapi uvicorn httpx

# ── Experiment tooling ──
pip install scipy statsmodels pandas numpy

# ── Azure AI Foundry SDK (Stage 3 Foundry latency comparison) ──
pip install azure-ai-inference azure-identity
```

> **Version pinning:** Before first benchmark run, freeze the environment with `pip freeze > requirements_frozen.txt` and log it as an MLflow artifact. Runtime version drift between Stage 1 and Stage 3 invalidates accuracy continuity. See Section 3.5.6 for the parity validation protocol.

### 3.4 Deployment Targets Comparison Path

The experiment validates three deployment targets for the SLM front-door, progressing from offline accuracy to production-ready latency:

| Deployment | Runtime | Latency Profile | Cost Profile | Experiment Stage |
|---|---|---|---|---|
| Local T4 (`transformers`) | `transformers` + `bitsandbytes` | Highest latency (no optimisation) | ~$0.75/hr fixed | Stage 1–2: accuracy baseline |
| Local T4 (`vLLM`) | vLLM OpenAI-compatible server | Lower latency (PagedAttention, continuous batching) | ~$0.75/hr fixed | Stage 3–4: self-hosted production path |
| Foundry serverless | Azure AI Foundry managed Phi-4-mini | Lowest latency (target), cold start risk | ~$0.13/1M input tokens | Stage 3–4: managed production path |

Stage 3 runs all three targets on the same corpus to produce a latency comparison that determines the production deployment topology. See Section 3.5 for full runtime specifications.

### 3.5 Inference Runtime Strategy

The experiment uses two distinct inference runtimes — one optimised for offline benchmark reproducibility, one for production-path latency validation. This is a deliberate split: Stage 1–2 need exact decoding control with zero server overhead; Stage 3–4 need an HTTP-serving runtime that matches the production API surface.

#### 3.5.1 Why Not Ollama

Ollama is ruled out for this experiment for three reasons:

1. **Quantisation mismatch.** Ollama wraps llama.cpp, which uses GGUF-format quantisation. The v1 benchmark used bitsandbytes 4-bit NF4 quantisation via HuggingFace `transformers`. Switching quantisation format changes model behaviour at the margin — accuracy and latency numbers from Ollama would not be comparable to v1 results. Reproducing the v1 baseline requires the same quantisation path.

2. **No concurrent request support.** Ollama's scheduler handles a few models on consumer hardware, not continuous batching or parallel request processing. Stage 3–4 need to measure latency under realistic concurrent load (the Plexor gateway may issue classify + score calls near-simultaneously). Ollama serialises these.

3. **Limited decoding control.** The locked benchmark contract requires exact parameter enforcement: `temperature=0.0`, `max_new_tokens=128` for classifier, `max_new_tokens=24` for scorer, greedy decoding with no sampling. Ollama exposes these through its API but does not guarantee identical decoding behaviour to `transformers` `generate()`, and the abstraction layer makes it harder to verify that the contract is held exactly.

#### 3.5.2 Runtime Selection by Stage

| Stage | Runtime | Quantisation | API Surface | Why |
|---|---|---|---|---|
| **1: Offline benchmark** | `transformers` + `bitsandbytes` | 4-bit NF4 | Direct Python `model.generate()` | Exact decoding control, v1 contract continuity, no server overhead in latency measurements |
| **2: Integration pipeline** | `transformers` + `bitsandbytes` | 4-bit NF4 | Direct Python (wrapped by pipeline classes) | Same accuracy baseline as Stage 1; pipeline tests classifier → scorer → router → hypervisor in-process |
| **3: Foundry latency** | `vLLM` (local) + Azure AI Foundry (remote) | NF4 (local) / Foundry-managed (remote) | OpenAI-compatible HTTP `/v1/chat/completions` | Production-path latency comparison; vLLM provides the local HTTP baseline that Foundry is measured against |
| **4: RCT** | `vLLM` (Arm B/C/D local) + Foundry (Arm B if latency passes) | Per-arm config | OpenAI-compatible HTTP | Plexor gateway calls the same HTTP API in RCT as in production; arms must be isolated at the endpoint level |

#### 3.5.3 Stage 1–2: `transformers` + `bitsandbytes` Direct Inference

This is the same stack used in the v1 RunPod A40 benchmark. Models are loaded one at a time into GPU memory, run through the full corpus, and unloaded before the next model loads. No inference server runs; latency is measured from `model.generate()` call to return.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Quantisation config — identical across all 4 models
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

def load_model(hf_repo: str):
    tokenizer = AutoTokenizer.from_pretrained(hf_repo, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        hf_repo,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2",  # if supported; fallback to eager
    )
    return model, tokenizer

def classify(model, tokenizer, prompt: str, system_prompt: str) -> str:
    """Locked contract: temperature=0.0, max_new_tokens=128, greedy."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    input_ids = tokenizer.apply_chat_template(
        messages, return_tensors="pt", add_generation_prompt=True
    ).to(model.device)

    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=128,
            temperature=None,     # greedy — no temperature scaling
            do_sample=False,      # deterministic
            top_p=None,
            repetition_penalty=1.0,
        )
    response = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True)
    return response

# Benchmark loop: load → run corpus → log → unload
for model_config in CANDIDATE_MODELS:
    model, tokenizer = load_model(model_config["hf_repo"])
    results = []
    for case in corpus:
        t0 = time.perf_counter()
        raw = classify(model, tokenizer, case["prompt"], CLASSIFIER_SYSTEM_PROMPT)
        latency_ms = (time.perf_counter() - t0) * 1000
        parsed = parse_json_safe(raw)  # JSON-safe parser with retry
        results.append({"case_id": case["id"], "raw": raw, "parsed": parsed, "latency_ms": latency_ms})
    log_to_mlflow(model_config["id"], results)
    del model, tokenizer
    torch.cuda.empty_cache()
```

**Key contract enforcement points:**

- `do_sample=False` ensures greedy decoding — no randomness
- `temperature=None` with `do_sample=False` prevents temperature scaling from affecting logits
- `max_new_tokens` is set per-task: 128 (classifier), 24 (scorer), 512 (compressor)
- `BitsAndBytesConfig` with `nf4` matches v1 quantisation exactly
- `trust_remote_code=True` required for both Phi and Qwen tokenizers
- Model loaded with `device_map="auto"` — on a single T4, this places everything on GPU 0

**Model swap procedure:** After each model completes the corpus, `del model, tokenizer` followed by `torch.cuda.empty_cache()` releases VRAM. The next model loads into clean GPU memory. Never run two models concurrently — the T4's 16 GB VRAM supports one ≤4B 4-bit model at a time with KV cache headroom.

#### 3.5.4 Stage 3–4: vLLM as OpenAI-Compatible Server

Once accuracy benchmarks are complete, Stage 3 shifts to production-path latency validation. vLLM serves the advancing models as OpenAI-compatible HTTP endpoints that the Plexor gateway can call with zero code changes — the same `POST /v1/chat/completions` shape used to call Anthropic, OpenAI, or Foundry.

**Why vLLM over SGLang:**

SGLang's RadixAttention gives higher peak throughput on large models (H100-class hardware, thousands of concurrent requests). For ≤4B models on a T4 at low-to-moderate concurrency (the Plexor front-door use case: 1–5 concurrent classify/score calls), the throughput advantage vanishes. vLLM wins on ecosystem maturity: broader model family support (both Phi and Qwen verified), more stable bitsandbytes integration, better documented OpenAI-compatible endpoint, and a larger community for troubleshooting. SGLang is worth revisiting if the production deployment moves to self-hosted on larger GPUs.

**Why vLLM over TGI:**

HuggingFace moved TGI to maintenance mode in December 2025. No new features, limited community investment. vLLM and SGLang are their recommended replacements for new deployments.

**Server configuration — Phi-4-mini-instruct:**

```bash
# Start vLLM server for Phi-4-mini
python -m vllm.entrypoints.openai.api_server \
    --model microsoft/Phi-4-mini-instruct \
    --quantization bitsandbytes \
    --load-format bitsandbytes \
    --dtype half \
    --max-model-len 4096 \
    --max-num-seqs 8 \
    --gpu-memory-utilization 0.85 \
    --port 8000 \
    --trust-remote-code \
    --enforce-eager  # disable CUDA graphs on T4 for stability
```

**Server configuration — Qwen2.5-3B-Instruct:**

```bash
# Start vLLM server for Qwen2.5-3B (swap — not concurrent with Phi)
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-3B-Instruct \
    --quantization bitsandbytes \
    --load-format bitsandbytes \
    --dtype half \
    --max-model-len 4096 \
    --max-num-seqs 8 \
    --gpu-memory-utilization 0.85 \
    --port 8000 \
    --trust-remote-code \
    --enforce-eager
```

**Client call — identical for both models:**

```python
import httpx

async def classify_via_vllm(prompt: str, system_prompt: str, port: int = 8000) -> dict:
    """Plexor gateway calls this same shape for Foundry, OpenAI, or local vLLM."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.post(
            f"http://localhost:{port}/v1/chat/completions",
            json={
                "model": "microsoft/Phi-4-mini-instruct",  # ignored by vLLM (single model)
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": 128,
                "temperature": 0.0,
            },
        )
    data = response.json()
    return data["choices"][0]["message"]["content"]
```

**Key vLLM configuration choices:**

| Parameter | Value | Rationale |
|---|---|---|
| `--quantization bitsandbytes` | NF4 | Matches Stage 1–2 quantisation for accuracy continuity |
| `--max-model-len 4096` | Tokens | Front-door prompts are short (classifier/scorer). 4096 is generous; saves VRAM for KV cache vs 32K default |
| `--max-num-seqs 8` | Concurrent | Plexor gateway issues ≤5 concurrent front-door calls. 8 provides headroom without exhausting T4 VRAM |
| `--gpu-memory-utilization 0.85` | Fraction | Leaves 15% VRAM for CUDA overhead. T4 16GB × 0.85 = 13.6 GB usable |
| `--enforce-eager` | Flag | Disables CUDA graph capture. T4's limited memory makes CUDA graphs fragile; eager mode is slower but stable |
| `--port 8000` | TCP | Plexor gateway default; matches Foundry endpoint pattern |

#### 3.5.5 Model Swap Protocol for RCT Arms

During the Stage 4 RCT, Arms B and C use different SLM models. Since the T4 cannot run both concurrently, the RCT uses one of two isolation strategies:

**Option 1: Sequential arm execution (simpler)**

Run all Arm B requests (Phi), stop the vLLM server, start with Qwen, run all Arm C requests. Arm D (hybrid split) runs last with a model-swap middleware that stops/starts vLLM per-request-family. This is operationally simpler but prevents interleaved arm execution.

**Option 2: Foundry + vLLM split (production-aligned)**

If Stage 3 validates Foundry latency: Arm B uses Foundry Phi-4-mini (remote, pay-per-token), Arm C uses local vLLM Qwen2.5-3B (T4), Arm D uses Foundry Phi + local vLLM Qwen. This matches the likely production topology where Phi runs on Foundry and Qwen runs self-hosted (Qwen has no Foundry endpoint).

**Decision point:** Stage 3 results determine which option is used. If Foundry latency passes, Option 2 is preferred because it validates the actual production deployment path during the RCT.

#### 3.5.6 Accuracy Continuity Validation

When transitioning from `transformers` (Stage 1–2) to vLLM (Stage 3–4), a validation pass is required to confirm that the serving runtime does not change model output:

```bash
# Run 20-case accuracy check: transformers vs vLLM on same model
python scripts/validate_runtime_parity.py \
    --model microsoft/Phi-4-mini-instruct \
    --corpus benchmark/progressive_test_cases_v2_120.jsonl \
    --n-cases 20 \
    --transformers-baseline results/stage1_phi4mini_baseline.json \
    --vllm-endpoint http://localhost:8000/v1/chat/completions \
    --output results/runtime_parity_check.json
```

Pass criteria: ≥95% exact output match (allowing for minor tokenisation differences at sequence boundaries). If parity fails, investigate vLLM quantisation config and fall back to `transformers`-based serving via FastAPI wrapper for Stage 3–4.

#### 3.5.7 Production Deployment Path

The inference runtime chosen for the experiment directly maps to production:

| Component | Experiment Runtime | Production Runtime | Migration |
|---|---|---|---|
| **Phi-4-mini classifier/scorer** | `transformers` (Stage 1–2) → vLLM (Stage 3–4) | Azure AI Foundry serverless | Zero code change — same OpenAI-compatible API |
| **Qwen2.5-3B classifier (if Arm C/D wins)** | `transformers` (Stage 1–2) → vLLM (Stage 3–4) | vLLM on dedicated NC4as_T4_v3 VM | Same vLLM config promoted from experiment |
| **Hypervisor deep eval** | `transformers` (all stages) | Azure AI Foundry (separate endpoint) | FoundryModelHandle wraps same API |
| **Compressor Stage 2** | `transformers` (all stages) | Same as classifier endpoint | Shares front-door endpoint |

The experiment validates the production path at every stage: Stage 1–2 prove accuracy, Stage 3 proves Foundry latency, Stage 4 proves the full gateway-to-model HTTP path under realistic traffic patterns.

---

## 4. Experiment Stages

### Stage 1: Locked Offline Pairwise Benchmark

**Goal:** Determine which models advance to integration testing. Reproduce v1 methodology with expanded corpus and upgraded model versions.

**Duration:** 2–3 days

**MLflow Experiment:** `plexor/slm-benchmark-v2`

**Inference Runtime:** `transformers` + `bitsandbytes` direct inference (Section 3.5.3). No inference server. Models loaded one at a time, run through full corpus, unloaded before next model. Latency measured from `model.generate()` call to return — no HTTP overhead.

#### 4.1.1 Corpus

Expand the v1 60-case corpus to 120 cases (20 per family) to improve family-level statistical power.

| Family | Cases | Source |
|---|---|---|
| code_simple | 20 | 10 carried from v1 + 10 new synthetic (GPT-4o seed, human reviewed) |
| code_complex | 20 | 10 carried from v1 + 10 new |
| cot_simple | 20 | 10 carried from v1 + 10 new |
| cot_complex | 20 | 10 carried from v1 + 10 new |
| hybrid_agentic | 20 | 10 carried from v1 + 10 new |
| hybrid_generative | 20 | 10 carried from v1 + 10 new |

Corpus file: `benchmark/progressive_test_cases_v2_120.jsonl`
Corpus SHA256 must be logged to MLflow before any benchmark run.

#### 4.1.2 Contract

| Parameter | Value |
|---|---|
| Classifier max_new_tokens | 128 |
| Scorer max_new_tokens | 24 |
| Compressor max_new_tokens | 512 |
| Quantisation | 4-bit NF4 (bitsandbytes) |
| Temperature | 0.0 (all tasks) |
| Decoding | Greedy (no sampling) |
| Parser | JSON-safe with retry (v1 contract) |

#### 4.1.3 Metrics Captured Per Run

| Metric | Type | MLflow Field |
|---|---|---|
| Exact label accuracy (overall) | Primary | `metrics/accuracy_overall` |
| Exact label accuracy (per family) | Primary | `metrics/accuracy_{family}` |
| Macro F1 | Primary | `metrics/f1_macro` |
| Agentic F1 | Primary | `metrics/f1_hybrid_agentic` |
| JSON parse success rate | Secondary | `metrics/parse_rate` |
| Mean latency (ms) | Secondary | `metrics/latency_mean_ms` |
| Median latency (ms) | Secondary | `metrics/latency_median_ms` |
| P95 latency (ms) | Secondary | `metrics/latency_p95_ms` |
| Compression ratio (if compressor tested) | Secondary | `metrics/compression_ratio` |
| Semantic similarity (if compressor tested) | Secondary | `metrics/semantic_similarity` |

#### 4.1.4 Statistical Tests

| Test | Comparison | Method |
|---|---|---|
| Overall accuracy | Pairwise (each Phi vs each Qwen, and within family) | Exact McNemar two-sided |
| Latency | Pairwise | Wilcoxon signed-rank |
| Parse reliability | Pairwise | Exact McNemar |
| Family-level accuracy | Per-family descriptive + Fisher exact where N allows | Fisher exact test |

Bonferroni correction applied for multiple comparisons (α/k where k = number of pairwise tests).

#### 4.1.5 Advancement Gate

A model advances to Stage 2 if it meets **any** of these criteria:

| Gate | Threshold |
|---|---|
| Overall accuracy | ≥0.55 (up from v1 Phi-3.5-mini's 0.5167) |
| F1 macro | ≥0.50 |
| Agentic F1 | ≥0.40 |
| Parse rate | ≥0.95 |
| Mean latency | ≤5,000ms on T4 (local) |

If no model passes all gates, the two best-performing models (one per family) advance with a documented exception.

#### 4.1.6 MLflow Logging

```python
# Example: Stage 1 benchmark run logging
with mlflow.start_run(experiment_id=benchmark_exp_id, run_name=f"{model_id}_120case"):
    mlflow.log_param("model_id", model_id)
    mlflow.log_param("corpus_sha256", corpus_hash)
    mlflow.log_param("corpus_size", 120)
    mlflow.log_param("contract", "jsonsafe_cls_tokens_128")
    mlflow.log_param("quant", "4bit_nf4")
    mlflow.log_param("gpu", "T4_16GB")
    mlflow.log_param("vm_size", "Standard_NC8as_T4_v3")
    mlflow.log_param("stage", "1_offline_benchmark")
    mlflow.log_param("runtime_engine", "transformers")       # NEW: track runtime
    mlflow.log_param("runtime_version", transformers.__version__)  # NEW: version pin
    mlflow.log_param("bitsandbytes_version", bitsandbytes.__version__)
    mlflow.log_param("torch_version", torch.__version__)

    # Frozen environment for reproducibility
    mlflow.log_artifact("requirements_frozen.txt")

    # Per-case results logged as artifact
    mlflow.log_artifact("results/per_case_results.json")

    # Aggregate metrics
    mlflow.log_metric("accuracy_overall", accuracy)
    mlflow.log_metric("f1_macro", f1_macro)
    mlflow.log_metric("f1_hybrid_agentic", agentic_f1)
    mlflow.log_metric("parse_rate", parse_rate)
    mlflow.log_metric("latency_mean_ms", latency_mean)
    mlflow.log_metric("latency_p95_ms", latency_p95)

    # Gate pass/fail
    mlflow.log_metric("gate_accuracy_pass", int(accuracy >= 0.55))
    mlflow.log_metric("gate_f1_macro_pass", int(f1_macro >= 0.50))
    mlflow.log_metric("gate_parse_pass", int(parse_rate >= 0.95))
    mlflow.log_metric("gate_latency_pass", int(latency_mean <= 5000))
```

---

### Stage 2: Integration Pipeline Validation

**Goal:** Validate the full Classify → Route → Supervise pipeline with the two advancing models. Test compressor, router, and hypervisor integration — not just classifier accuracy.

**Duration:** 3–5 days

**MLflow Experiment:** `plexor/slm-integration-v2`

**Inference Runtime:** `transformers` + `bitsandbytes` direct inference (Section 3.5.3). Same as Stage 1 — pipeline components call `model.generate()` directly via Python wrapper classes (FoundryModelHandle stub). No HTTP server; end-to-end pipeline latency measured in-process.

**Prerequisite:** Stage 1 complete; two models selected.

#### 4.2.1 Test Corpus

The EIR acceptance test scenarios provide real-world validation alongside the synthetic benchmark:

| Corpus | Cases | Purpose |
|---|---|---|
| Progressive 120 (from Stage 1) | 120 | Full pipeline classification + routing accuracy |
| MBPP coding challenges | 25 | Real code-gen tasks through full pipeline |
| EIR Scenario 2 variants | 10 | Claude Code plugin integration path |
| Compression stress tests | 30 | Verbose prompts with known compression targets |
| Hypervisor challenge set | 40 | 50/50 clean vs flagged responses, all 6 watch-point types |

Total: ~225 test cases.

#### 4.2.2 Pipeline Components Under Test

| Component | What Is Validated | Pass Criteria |
|---|---|---|
| **Compressor Stage 1** (rule-based) | Token reduction on verbose prompts | Ratio ≤0.80 on ≥70% of verbose prompts |
| **Compressor Stage 2** (SLM) | Semantic similarity after LLM compression | Similarity ≥0.88 on all accepted compressions |
| **Classifier** | Label accuracy against ground truth | ≥0.55 overall; ≥0.40 agentic F1 |
| **Scorer** | Score within expected range for known cases | 100% of scores in [1.0, 10.0]; fallback fires correctly |
| **Router** | Deterministic tier assignment | 100% agreement with routing table on (label, score) |
| **Router invariant** | hybrid/agentic → frontier always | Zero violations across all hybrid/agentic cases |
| **Hypervisor Stage 1** (regex) | Watch-point detection recall | plan_mode_leak ≥0.85; fake_git_sha = 1.00; unmocked_tests ≥0.90 |
| **Hypervisor Stage 2** (SLM) | Quality score correlation with human ratings | Spearman ≥0.75 on 50-response calibration set |
| **End-to-end latency** | Full pipeline P95 | ≤200ms front-door (classify + score + compress) |
| **Fallback behaviour** | Graceful degradation on model timeout/parse failure | 100% of simulated failures produce valid fallback |

#### 4.2.3 Compression Evaluation Detail

The v1 MBPP case study showed modest compression (9–16% token reduction). Stage 2 evaluates whether the SLM compressor can hit the spec's ≥30% target:

| Prompt Type | Expected Compression | Measurement |
|---|---|---|
| Meeting-context preamble + task | ≥40% reduction | Strip preamble, preserve task |
| Verbose code review request | ≥25% reduction | Preserve all file paths, function names, constraints |
| Clean technical prompt (no bloat) | <10% reduction (should skip Stage 2) | Stage 2 must NOT fire |
| Multi-paragraph architecture request | ≥20% reduction | Preserve all technical terms and numeric values |

#### 4.2.4 Routing Accuracy Evaluation

For each test case, compare the SLM-assigned (label, score) → tier against the ground-truth tier:

```
routing_accuracy = count(predicted_tier == ground_truth_tier) / total_cases
```

Additionally measure **cost impact**: for each mis-route, compute the cost delta (over-route costs more, under-route risks quality). Log to MLflow as `metrics/routing_cost_delta_usd`.

#### 4.2.5 MLflow Logging

```python
with mlflow.start_run(experiment_id=integration_exp_id, run_name=f"{model_id}_pipeline"):
    mlflow.log_param("stage", "2_integration")
    mlflow.log_param("model_id", model_id)
    mlflow.log_param("pipeline_version", pipeline_version)
    mlflow.log_param("corpus_cases", 225)

    # Pipeline-level metrics
    mlflow.log_metric("routing_accuracy", routing_acc)
    mlflow.log_metric("routing_cost_delta_usd", cost_delta)
    mlflow.log_metric("compression_ratio_avg", avg_compression)
    mlflow.log_metric("compression_similarity_avg", avg_similarity)
    mlflow.log_metric("hypervisor_precision", hv_precision)
    mlflow.log_metric("hypervisor_recall", hv_recall)
    mlflow.log_metric("hypervisor_fpr", hv_fpr)
    mlflow.log_metric("pipeline_p95_ms", pipeline_p95)

    # Per-component latency
    mlflow.log_metric("latency_compress_p95_ms", compress_p95)
    mlflow.log_metric("latency_classify_p95_ms", classify_p95)
    mlflow.log_metric("latency_score_p95_ms", score_p95)
    mlflow.log_metric("latency_hypervisor_p95_ms", hv_p95)

    # Gate checks
    mlflow.log_metric("gate_routing_accuracy_pass", int(routing_acc >= 0.85))
    mlflow.log_metric("gate_compression_similarity_pass", int(avg_similarity >= 0.88))
    mlflow.log_metric("gate_hypervisor_recall_pass", int(hv_recall >= 0.75))
    mlflow.log_metric("gate_latency_sla_pass", int(pipeline_p95 <= 200))
    mlflow.log_metric("gate_all_green", int(all_gates_pass))
```

---

### Stage 3: Foundry & vLLM Latency Comparison

**Goal:** Validate that production-path serving runtimes meet the ≤200ms P95 latency target. Compare three deployment targets: local `transformers` (Stage 1–2 baseline), local vLLM (self-hosted production path), and Azure AI Foundry serverless (managed production path).

**Duration:** 1–2 days

**MLflow Experiment:** `plexor/foundry-latency-v2`

**Inference Runtime:** vLLM OpenAI-compatible server (Section 3.5.4) for local serving; Azure AI Foundry serverless for remote. Runtime parity validation (Section 3.5.6) runs first to confirm vLLM does not change model output vs `transformers`.

**Prerequisite:** Stage 2 complete; pipeline validated on local GPU. Runtime parity check passed (≥95% exact output match between `transformers` and vLLM on 20-case subset).

#### 4.3.1 Test Protocol

Run the same 120-case classification corpus against three deployment targets:

| Target | Endpoint | Runtime | Warmup | Measurement Window |
|---|---|---|---|---|
| Local transformers | In-process `model.generate()` | `transformers` + `bitsandbytes` | 3 warmup calls | 120 cases × 3 runs |
| Local vLLM | `http://localhost:8000/v1/chat/completions` | vLLM (Section 3.5.4 config) | 5 warmup requests | 120 cases × 3 runs |
| Foundry serverless | `plexor-frontdoor-phi4` (eastus) | Azure AI Foundry managed | 5 warmup requests + keep-alive ping | 120 cases × 3 runs |

Pre-warm the Foundry endpoint with 5 requests before timing begins. Log cold-start latency separately. For vLLM, wait until server reports "ready" before timing.

#### 4.3.2 Metrics

| Metric | Local transformers | Local vLLM Target | Foundry Target |
|---|---|---|---|
| Classify P50 (ms) | Measured (baseline) | ≤60ms | ≤40ms |
| Classify P95 (ms) | Measured (baseline) | ≤150ms | ≤100ms |
| Classify + Score P95 (ms) | Measured (baseline) | ≤250ms | ≤200ms |
| Cold start (ms) | N/A | ≤5,000ms (first request after server start) | ≤2,000ms |
| Accuracy (must be identical) | Baseline | Within ±0.02 of baseline | Within ±0.02 of baseline |
| Throughput (req/min) | Measured | ≥50 sustained | ≥50 sustained |

#### 4.3.3 vLLM-Specific Measurements

In addition to classification latency, measure vLLM operational metrics:

| Metric | Method | Pass Criteria |
|---|---|---|
| Server startup time | Time from `python -m vllm.entrypoints...` to first successful `/health` | ≤60s |
| Memory utilisation | `nvidia-smi` during sustained load | ≤14 GB (85% of T4's 16 GB) |
| Request queue depth | vLLM `/metrics` endpoint | ≤3 at 50 req/min sustained |
| Error rate | HTTP 5xx / total requests | 0% on 120-case corpus |

#### 4.3.4 Decision Matrix

| Foundry Latency | vLLM Latency | Decision |
|---|---|---|
| ✅ Passes (≤200ms P95) | ✅ Passes | Use Foundry for Phi (production), vLLM for Qwen (no Foundry endpoint). RCT Option 2 (Section 3.5.5). |
| ✅ Passes | ❌ Fails | Use Foundry for both Phi arms. Qwen arm uses Foundry if Qwen endpoint available, else vLLM with tuning. |
| ❌ Fails | ✅ Passes | Self-host both models on vLLM. Escalate Foundry to provisioned throughput for production. |
| ❌ Fails | ❌ Fails | Escalate: investigate `--enable-chunked-prefill`, `--speculative-model`, or upgrade to NC16as_T4_v3 (2× T4). |

---

### Stage 4: Randomised Controlled Trial (RCT)

**Goal:** Measure causal treatment effects of the SLM front-door on cost, quality, and latency against the control (no routing layer).

**Duration:** 1–2 weeks

**MLflow Experiments:** `plexor/rct-arm-A`, `plexor/rct-arm-B`, `plexor/rct-arm-C`, `plexor/rct-arm-D`

**Inference Runtime:** vLLM OpenAI-compatible server for local SLM serving (Section 3.5.4); Azure AI Foundry for Phi if Stage 3 latency passes. Arm isolation strategy selected per Section 3.5.5 based on Stage 3 results. All arms receive requests via the same HTTP API shape (`/v1/chat/completions`) — the Plexor gateway makes no code-level distinction between vLLM, Foundry, or direct Anthropic calls.

**Prerequisite:** Stage 3 complete; latency SLA validated; `gate_all_green = 1` from Stage 2. Runtime parity check (Section 3.5.6) passed.

#### 4.4.1 Arm Definitions

| Arm | Label | Description | SLM Model | Routing | Hypervisor |
|---|---|---|---|---|---|
| **A** | Control | model="auto", direct to provider | None | None | None |
| **B** | Phi pretrained | Full pipeline, base Phi-4-mini-instruct | Phi-4-mini | Active | Active |
| **C** | Qwen pretrained | Full pipeline, base Qwen2.5-3B-Instruct | Qwen2.5-3B | Active | Active |
| **D** | Hybrid split | Phi for code/CoT, Qwen for hybrid_agentic | Both | Active (split) | Active |

> **Note:** The v1 design had Arms A/B/C with Arm C as fine-tuned Phi. This v2 design replaces Arm C with a Qwen arm and adds Arm D (hybrid split) because the v1 benchmark showed model-family-level complementarity. Fine-tuning becomes Stage 5 after the RCT identifies the winning policy.

#### 4.4.2 Arm Isolation (Critical)

The v1 RCT readiness cycle identified arm URL aliasing as an invalidating confound. Stage 4 requires **distinct backend endpoints per arm.** The exact topology depends on Stage 3 results (see Section 3.5.5):

**If Foundry latency passes (Option 2 — production-aligned):**

| Arm | SLM Endpoint | Serving Runtime | Downstream Provider |
|---|---|---|---|
| A | None (no front-door) | N/A | Direct to Anthropic API |
| B | Foundry `plexor-frontdoor-phi4` (remote) | Azure AI Foundry serverless | Plexor gateway → tier provider |
| C | `http://localhost:8000` (local T4) | vLLM serving Qwen2.5-3B | Plexor gateway → tier provider |
| D | Foundry Phi (code/CoT) + local vLLM Qwen (hybrid_agentic) | Foundry + vLLM (split) | Plexor gateway → tier provider |

**If Foundry latency fails (Option 1 — sequential execution):**

| Arm | SLM Endpoint | Serving Runtime | Execution |
|---|---|---|---|
| A | None | N/A | Run first (control baseline) |
| B | `http://localhost:8000` | vLLM serving Phi-4-mini | Run second (stop after completion) |
| C | `http://localhost:8000` | vLLM serving Qwen2.5-3B | Run third (swap model, restart vLLM) |
| D | `http://localhost:8000` | vLLM serving Phi, then swap to Qwen per-family | Run last (model-swap middleware) |

Traffic split via Azure Container Apps revision-based routing (25% per arm) or session-hash assignment (SHA-256 mod 4).

#### 4.4.3 Randomisation

```python
import hashlib

def assign_arm(session_id: str, n_arms: int = 4) -> str:
    h = hashlib.sha256(session_id.encode()).hexdigest()
    arm_index = int(h, 16) % n_arms
    return ["A", "B", "C", "D"][arm_index]
```

Session-level assignment (not request-level) prevents mid-session arm switches.

#### 4.4.4 Primary Hypotheses

| ID | Hypothesis | Metric | Expected Direction |
|---|---|---|---|
| H1 | Routing reduces cost | Avg USD per request | B, C, D < A |
| H2 | Compression reduces token spend | Avg input tokens sent downstream | B, C, D < A |
| H3 | Hypervisor reduces defect rate | % responses with ≥1 watch-point flag | B, C, D < A |
| H4 | Front-door latency is acceptable | Pipeline P95 overhead (ms) | B, C, D ≤ 200ms |
| H5 | Hybrid split outperforms single-model | Routing accuracy (ground truth) | D > B and D > C |
| H6 | Quality is maintained | Human-rated overall quality score | B, C, D ≥ A |

#### 4.4.5 Sample Size

Target N = 250 per arm (1,000 total). Power analysis: 80% power to detect a 10-point absolute improvement in routing accuracy at α = 0.05, Bonferroni-corrected (α/6 = 0.0083 per hypothesis).

Estimated wall-clock time at 50 req/min sustained: ~5 hours per arm.

#### 4.4.6 Metrics Logged Per Request

```python
# Logged to MLflow via RCTMetricsBuffer (flushes every 10 requests)
{
    "request_id": str,
    "arm": str,
    "session_id": str,
    "task_label": str,           # Null for Arm A
    "confidence": float,         # Null for Arm A
    "complexity_score": float,   # Null for Arm A
    "tier_assigned": str,        # Null for Arm A
    "compression_ratio": float,  # 1.0 for Arm A (no compression)
    "watch_points": list[str],   # Empty for Arm A
    "quality_scores": dict,      # Null for Arm A
    "downstream_model": str,
    "downstream_cost_usd": float,
    "frontend_runtime": str,     # "foundry" | "vllm" | null (Arm A)
    "frontend_model": str,       # "phi-4-mini-instruct" | "qwen2.5-3b-instruct" | null
    "front_door_latency_ms": float,  # 0.0 for Arm A
    "downstream_latency_ms": float,
    "total_latency_ms": float,
    "human_quality_label": str,  # Added post-hoc on 50-sample subset
}
```

#### 4.4.7 RCT Analysis Script

```bash
# Run after all arms reach N ≥ 250
python scripts/rct_analysis.py \
    --experiment-prefix plexor/rct-arm \
    --min-n 250 \
    --alpha 0.05 \
    --correction bonferroni \
    --output results/rct_analysis_v2.json
```

Output includes per-hypothesis test results, effect sizes, confidence intervals, and a promotion recommendation.

---

## 5. MLflow Configuration

### 5.1 Server Setup

```bash
# PostgreSQL backend (on the VM or Azure PostgreSQL Flexible)
export MLFLOW_BACKEND_STORE_URI="postgresql://plexor:${PG_PASSWORD}@plexor-pg.postgres.database.azure.com:5432/mlflow"

# Blob storage for artifacts
export MLFLOW_ARTIFACT_ROOT="wasbs://mlflow-artifacts@plexorstorage.blob.core.windows.net/"

# Start MLflow server
mlflow server \
    --backend-store-uri $MLFLOW_BACKEND_STORE_URI \
    --default-artifact-root $MLFLOW_ARTIFACT_ROOT \
    --host 0.0.0.0 \
    --port 5000
```

### 5.2 Experiment Registry

| Experiment Name | Stage | Description |
|---|---|---|
| `plexor/slm-benchmark-v2` | 1 | Offline pairwise model comparison (4 models × 120 cases) |
| `plexor/slm-integration-v2` | 2 | Full pipeline validation (2 models × 225 cases) |
| `plexor/foundry-latency-v2` | 3 | Foundry vs local latency comparison |
| `plexor/rct-arm-A` | 4 | RCT control arm (no routing) |
| `plexor/rct-arm-B` | 4 | RCT Phi pretrained arm |
| `plexor/rct-arm-C` | 4 | RCT Qwen pretrained arm |
| `plexor/rct-arm-D` | 4 | RCT hybrid split arm |
| `plexor/rct-analysis-v2` | 4 | Cross-arm analysis and promotion decision |
| `plexor/slm-finetune-v1` | 5 (future) | Fine-tuning on winning model (post-RCT) |

### 5.3 Run Naming Convention

```
{stage}_{model_id}_{corpus}_{timestamp}
```

Examples:
- `s1_phi4mini_120case_20260315T1430`
- `s2_qwen25_3b_pipeline_20260318T0900`
- `s4_armB_rct_20260325T1200`

### 5.4 Artifact Storage

Each run logs the following artifacts:

| Artifact | Format | Logged At |
|---|---|---|
| Per-case results | JSON | All stages |
| Significance report | JSON | Stage 1, 4 |
| Confusion matrix | PNG | Stage 1, 2 |
| Latency histogram | PNG | All stages |
| Routing table used | JSON | Stage 2, 4 |
| Corpus file (hash only) | Text | All stages |
| Pipeline config snapshot | YAML | Stage 2, 3, 4 |
| Hypervisor calibration set | JSON | Stage 2 |

---

## 6. Test Case Design

### 6.1 Progressive Benchmark Test Cases (120 cases)

#### code_simple (20 cases)

| TC# | Prompt Summary | Ground Truth Label | Expected Score | Expected Tier |
|---|---|---|---|---|
| CS-01 | Write `normalize_whitespace()` | code/simple | 2.0–3.0 | fast |
| CS-02 | FizzBuzz implementation | code/simple | 1.5–2.5 | fast |
| CS-03 | Single unittest for `add(a,b)` | code/simple | 1.0–2.0 | fast |
| CS-04 | Palindrome checker function | code/simple | 2.0–3.0 | fast |
| CS-05 | CSV row counter script | code/simple | 2.5–3.5 | fast |
| CS-06 | String reversal with edge cases | code/simple | 1.5–2.5 | fast |
| CS-07 | Temperature converter (C↔F) | code/simple | 1.0–2.0 | fast |
| CS-08 | `test_duplicate(nums)` (EIR S2) | code/simple | 2.0–3.0 | fast |
| CS-09 | JSON key extractor one-liner | code/simple | 2.0–3.5 | fast |
| CS-10 | Fibonacci up to N | code/simple | 2.0–3.0 | fast |
| CS-11 | List deduplication preserving order | code/simple | 2.5–3.5 | fast |
| CS-12 | Vowel counter function | code/simple | 1.0–2.0 | fast |
| CS-13 | Binary search implementation | code/simple | 3.0–4.0 | fast |
| CS-14 | Date formatter (ISO→US) | code/simple | 2.5–3.5 | fast |
| CS-15 | Email regex validator | code/simple | 3.0–4.0 | fast |
| CS-16 | Read file, count words | code/simple | 2.5–3.5 | fast |
| CS-17 | Flatten nested list | code/simple | 3.0–4.0 | fast |
| CS-18 | ROT13 cipher | code/simple | 2.0–3.0 | fast |
| CS-19 | Matrix transpose | code/simple | 2.5–3.0 | fast |
| CS-20 | Argparse hello-world CLI | code/simple | 3.0–4.0 | fast |

#### code_complex (20 cases)

| TC# | Prompt Summary | Ground Truth Label | Expected Score | Expected Tier |
|---|---|---|---|---|
| CC-01 | REST API client with retry + auth | code/complex | 6.0–7.5 | mid–frontier |
| CC-02 | Config validator with schema + custom exceptions | code/complex | 5.5–7.0 | mid |
| CC-03 | CLI app with SQLite persistence | code/complex | 7.0–8.0 | frontier |
| CC-04 | Async job queue with worker pool | code/complex | 7.5–8.5 | frontier |
| CC-05 | Multi-file Python package with `__init__.py` exports | code/complex | 5.0–6.5 | mid |
| CC-06 | Rate limiter middleware (token bucket) | code/complex | 6.0–7.0 | mid |
| CC-07 | S3-compatible file uploader with multipart | code/complex | 7.0–8.0 | frontier |
| CC-08 | JWT auth module with refresh token rotation | code/complex | 7.0–8.5 | frontier |
| CC-09 | Database migration runner (up/down) | code/complex | 6.5–7.5 | mid–frontier |
| CC-10 | WebSocket chat server with rooms | code/complex | 7.5–8.5 | frontier |
| CC-11 | CI/CD pipeline YAML generator | code/complex | 5.5–6.5 | mid |
| CC-12 | Prometheus metrics exporter library | code/complex | 6.0–7.0 | mid |
| CC-13 | GraphQL schema + resolver for user CRUD | code/complex | 7.0–8.0 | frontier |
| CC-14 | Custom logging handler with rotation + JSON | code/complex | 5.0–6.0 | mid |
| CC-15 | OAuth2 client credentials flow wrapper | code/complex | 6.5–7.5 | mid–frontier |
| CC-16 | Redis-backed session store | code/complex | 6.0–7.0 | mid |
| CC-17 | Event-driven state machine | code/complex | 7.0–8.0 | frontier |
| CC-18 | PDF report generator with tables + charts | code/complex | 6.5–7.5 | mid–frontier |
| CC-19 | gRPC service definition + Python server | code/complex | 7.5–8.5 | frontier |
| CC-20 | Plugin architecture with dynamic loading | code/complex | 7.0–8.0 | frontier |

#### cot_simple (20 cases)

| TC# | Prompt Summary | Ground Truth Label | Expected Score | Expected Tier |
|---|---|---|---|---|
| TS-01 | Explain JWT vs session-based auth | CoT/simple | 2.0–3.5 | fast |
| TS-02 | Compare REST vs GraphQL | CoT/simple | 2.5–4.0 | fast |
| TS-03 | What is dependency injection? | CoT/simple | 1.5–3.0 | fast |
| TS-04 | Explain CAP theorem simply | CoT/simple | 3.0–4.0 | fast |
| TS-05 | Pros/cons of microservices | CoT/simple | 3.0–4.0 | fast |
| TS-06 | When to use NoSQL vs SQL? | CoT/simple | 2.5–3.5 | fast |
| TS-07 | What is eventual consistency? | CoT/simple | 2.0–3.0 | fast |
| TS-08 | Explain CORS in plain English | CoT/simple | 1.5–2.5 | fast |
| TS-09 | TCP vs UDP comparison | CoT/simple | 2.0–3.0 | fast |
| TS-10 | What are database indexes and when to use them? | CoT/simple | 2.5–3.5 | fast |
| TS-11 | Explain Docker vs VMs | CoT/simple | 2.0–3.0 | fast |
| TS-12 | What is a load balancer? | CoT/simple | 1.5–2.5 | fast |
| TS-13 | Explain Git rebase vs merge | CoT/simple | 2.5–3.5 | fast |
| TS-14 | What is ACID in databases? | CoT/simple | 2.0–3.0 | fast |
| TS-15 | OAuth2 flow overview | CoT/simple | 3.0–4.0 | fast |
| TS-16 | What is idempotency in APIs? | CoT/simple | 2.0–3.0 | fast |
| TS-17 | Explain pub/sub messaging pattern | CoT/simple | 2.5–3.5 | fast |
| TS-18 | Monorepo vs polyrepo tradeoffs | CoT/simple | 3.0–4.0 | fast |
| TS-19 | What is a reverse proxy? | CoT/simple | 1.5–2.5 | fast |
| TS-20 | Explain blue-green deployment | CoT/simple | 2.5–3.5 | fast |

#### cot_complex (20 cases)

| TC# | Prompt Summary | Ground Truth Label | Expected Score | Expected Tier |
|---|---|---|---|---|
| TC-01 | Design memo: event-driven order processing system | CoT/complex | 7.0–8.5 | frontier |
| TC-02 | Architecture review: monolith → microservices migration | CoT/complex | 7.5–9.0 | frontier |
| TC-03 | Risk analysis: moving from on-prem to cloud-native | CoT/complex | 6.5–8.0 | frontier |
| TC-04 | Trade-off analysis: Kafka vs RabbitMQ vs SQS | CoT/complex | 6.0–7.5 | frontier |
| TC-05 | Security audit report template for REST API | CoT/complex | 6.5–7.5 | frontier |
| TC-06 | Multi-stakeholder capacity planning memo | CoT/complex | 7.0–8.0 | frontier |
| TC-07 | Incident post-mortem template with timeline analysis | CoT/complex | 6.0–7.0 | mid–frontier |
| TC-08 | Data modelling decision: relational vs graph for social network | CoT/complex | 7.0–8.0 | frontier |
| TC-09 | API versioning strategy document | CoT/complex | 5.5–7.0 | mid–frontier |
| TC-10 | Performance optimisation plan for high-traffic e-commerce | CoT/complex | 7.5–8.5 | frontier |
| TC-11 | Compliance review: GDPR data processing architecture | CoT/complex | 7.0–8.5 | frontier |
| TC-12 | Technical debt prioritisation framework | CoT/complex | 6.0–7.5 | frontier |
| TC-13 | Observability strategy: logs vs metrics vs traces | CoT/complex | 6.5–7.5 | frontier |
| TC-14 | Database sharding strategy for multi-tenant SaaS | CoT/complex | 7.5–9.0 | frontier |
| TC-15 | Disaster recovery plan for distributed system | CoT/complex | 7.0–8.5 | frontier |
| TC-16 | Cost modelling: serverless vs container-based deployment | CoT/complex | 6.5–8.0 | frontier |
| TC-17 | Team topology analysis for platform engineering | CoT/complex | 6.0–7.5 | frontier |
| TC-18 | ML pipeline architecture review | CoT/complex | 7.5–9.0 | frontier |
| TC-19 | Zero-trust security architecture design | CoT/complex | 7.5–8.5 | frontier |
| TC-20 | SLA/SLO definition for multi-service platform | CoT/complex | 6.5–8.0 | frontier |

#### hybrid_agentic (20 cases)

| TC# | Prompt Summary | Ground Truth Label | Expected Score | Expected Tier |
|---|---|---|---|---|
| HA-01 | Build CLI from scratch + run tests + git commit + verify SHA | hybrid/agentic | 9.0–10.0 | frontier ★ |
| HA-02 | Scaffold FastAPI project + Dockerfile + Makefile + smoke test | hybrid/agentic | 8.5–9.5 | frontier ★ |
| HA-03 | Create Python package + publish to test PyPI + verify install | hybrid/agentic | 9.0–10.0 | frontier ★ |
| HA-04 | Debug failing CI pipeline + fix + push + verify green | hybrid/agentic | 8.0–9.5 | frontier ★ |
| HA-05 | Multi-file refactor: extract class + update imports + run tests | hybrid/agentic | 8.0–9.0 | frontier ★ |
| HA-06 | URL shortener: build + test + deploy locally + verify E2E | hybrid/agentic | 9.0–10.0 | frontier ★ |
| HA-07 | Database migration: write schema + migrate + seed + verify | hybrid/agentic | 8.5–9.5 | frontier ★ |
| HA-08 | Set up monitoring: Prometheus + Grafana + alert rules + verify | hybrid/agentic | 8.5–9.5 | frontier ★ |
| HA-09 | Autonomous debugger: reproduce bug → root cause → fix → test | hybrid/agentic | 9.0–10.0 | frontier ★ |
| HA-10 | Build REST API + OpenAPI spec + client SDK + integration tests | hybrid/agentic | 9.5–10.0 | frontier ★ |
| HA-11 | Create Terraform module + plan + apply + verify resources | hybrid/agentic | 8.5–9.5 | frontier ★ |
| HA-12 | Write + run load test suite + generate report + recommendations | hybrid/agentic | 8.0–9.0 | frontier ★ |
| HA-13 | Implement feature flag system + toggle + verify behaviour | hybrid/agentic | 8.0–9.0 | frontier ★ |
| HA-14 | Build data pipeline: ingest → transform → validate → output | hybrid/agentic | 8.5–9.5 | frontier ★ |
| HA-15 | Security scan: run SAST + triage findings + fix critical + re-scan | hybrid/agentic | 8.5–9.5 | frontier ★ |
| HA-16 | Create GitHub Actions workflow + test matrix + badge + verify | hybrid/agentic | 8.0–9.0 | frontier ★ |
| HA-17 | Build + test + containerise + push to registry + verify pull | hybrid/agentic | 9.0–10.0 | frontier ★ |
| HA-18 | Full TDD cycle: write tests first → implement → refactor → green | hybrid/agentic | 8.5–9.5 | frontier ★ |
| HA-19 | Multi-service integration: service A calls B, verify end-to-end | hybrid/agentic | 9.0–10.0 | frontier ★ |
| HA-20 | Automated code review: clone → analyse → report → suggest fixes | hybrid/agentic | 8.0–9.0 | frontier ★ |

★ hybrid/agentic always routes to frontier regardless of score (hard invariant).

#### hybrid_generative (20 cases)

| TC# | Prompt Summary | Ground Truth Label | Expected Score | Expected Tier |
|---|---|---|---|---|
| HG-01 | Technical blog post with working code examples | hybrid/generative | 5.5–7.0 | mid |
| HG-02 | Tutorial: build a todo app with annotated code | hybrid/generative | 6.0–7.5 | mid–frontier |
| HG-03 | README.md with install instructions + usage examples | hybrid/generative | 4.5–6.0 | mid |
| HG-04 | API documentation with curl examples | hybrid/generative | 5.0–6.5 | mid |
| HG-05 | Conference talk outline with demo code | hybrid/generative | 6.0–7.0 | mid |
| HG-06 | Onboarding guide with setup scripts | hybrid/generative | 5.5–7.0 | mid |
| HG-07 | Technical newsletter issue with code snippets | hybrid/generative | 5.0–6.5 | mid |
| HG-08 | Book chapter: intro to async Python with examples | hybrid/generative | 6.5–8.0 | mid–frontier |
| HG-09 | Comparison article: 3 frameworks with benchmark code | hybrid/generative | 6.5–8.0 | mid–frontier |
| HG-10 | Internal wiki page: deployment runbook with scripts | hybrid/generative | 5.0–6.5 | mid |
| HG-11 | Developer changelog with migration code | hybrid/generative | 5.0–6.0 | mid |
| HG-12 | Jupyter notebook narrative: data analysis walkthrough | hybrid/generative | 6.0–7.5 | mid–frontier |
| HG-13 | Workshop material: exercises + solutions | hybrid/generative | 6.5–7.5 | mid–frontier |
| HG-14 | Technical RFC with reference implementation | hybrid/generative | 7.0–8.5 | frontier |
| HG-15 | Video script with code walkthroughs | hybrid/generative | 5.5–7.0 | mid |
| HG-16 | Product release notes with code examples | hybrid/generative | 4.5–6.0 | mid |
| HG-17 | Teaching material: design patterns with implementations | hybrid/generative | 6.5–8.0 | mid–frontier |
| HG-18 | Open-source contribution guide with PR walkthrough | hybrid/generative | 5.5–7.0 | mid |
| HG-19 | Technical proposal with proof-of-concept code | hybrid/generative | 7.0–8.5 | frontier |
| HG-20 | Case study: performance optimisation with before/after code | hybrid/generative | 6.0–7.5 | mid–frontier |

### 6.2 Hypervisor Challenge Set (40 cases)

| Category | Clean Cases | Flagged Cases | Watch Points Tested |
|---|---|---|---|
| plan_mode_leak | 5 | 5 | Response begins with "I will...", "Let me plan...", "Step 1:" |
| fake_git_sha | 5 | 5 | Fabricated 40-char hex strings in code output |
| unmocked_tests | 5 | 5 | Test files with `open()`, `requests.get()`, `boto3.` without mocking |
| incomplete_output | 5 | 5 | Missing STATUS block, truncated mid-sentence, absent declared artifacts |

### 6.3 Compression Stress Tests (30 cases)

| Category | Cases | Expected Compression | Key Validation |
|---|---|---|---|
| Meeting-context bloat | 10 | ≥40% | Preamble removed, task preserved |
| Verbose repetitive instructions | 10 | ≥30% | Deduplicated, constraints preserved |
| Already-concise technical prompts | 10 | <10% (Stage 2 should NOT fire) | No unnecessary compression |

---

## 7. Configuration Files

### 7.1 Router Config (`config/router_config.json`)

```json
{
  "version": "2.0",
  "routing_table": {
    "code/simple":        {"low": "fast", "mid": "fast",     "high": "mid",      "very_high": "mid"},
    "code/complex":       {"low": "mid",  "mid": "mid",      "high": "frontier", "very_high": "frontier"},
    "CoT/simple":         {"low": "fast", "mid": "fast",     "high": "mid",      "very_high": "mid"},
    "CoT/complex":        {"low": "mid",  "mid": "frontier", "high": "frontier", "very_high": "frontier"},
    "hybrid/agentic":     {"low": "frontier", "mid": "frontier", "high": "frontier", "very_high": "frontier"},
    "hybrid/generative":  {"low": "mid",  "mid": "mid",      "high": "frontier", "very_high": "frontier"}
  },
  "score_buckets": {
    "low":       [1.0, 4.0],
    "mid":       [4.1, 7.0],
    "high":      [7.1, 9.0],
    "very_high": [9.1, 10.0]
  },
  "invariants": {
    "hybrid/agentic": "always_frontier"
  }
}
```

### 7.2 Benchmark Config (`config/benchmark_config.yaml`)

```yaml
benchmark:
  version: "2.0"
  corpus: "benchmark/progressive_test_cases_v2_120.jsonl"
  contract: "jsonsafe_cls_tokens_128"

runtime:
  stage1_2:
    engine: "transformers"
    quantisation_backend: "bitsandbytes"
    quantisation_config:
      load_in_4bit: true
      bnb_4bit_quant_type: "nf4"
      bnb_4bit_compute_dtype: "float16"
      bnb_4bit_use_double_quant: true
    device_map: "auto"
    attn_implementation: "flash_attention_2"  # fallback: eager
    trust_remote_code: true

  stage3_4:
    engine: "vllm"
    quantisation: "bitsandbytes"
    load_format: "bitsandbytes"
    dtype: "half"
    max_model_len: 4096
    max_num_seqs: 8
    gpu_memory_utilisation: 0.85
    enforce_eager: true  # disable CUDA graphs on T4
    port: 8000
    api_format: "openai_compatible"  # /v1/chat/completions

  foundry:
    engine: "azure_ai_foundry"
    sdk: "azure-ai-inference"
    endpoint_env: "FOUNDRY_FRONTDOOR_URL"
    key_env: "FOUNDRY_FRONTDOOR_KEY"
    model: "Phi-4-mini-instruct"

models:
  - id: "phi-4-mini-instruct"
    hf_repo: "microsoft/Phi-4-mini-instruct"
    quant: "4bit_nf4"
    max_new_tokens_classify: 128
    max_new_tokens_score: 24
    max_new_tokens_compress: 512
    temperature: 0.0

  - id: "phi-3.5-mini-instruct"
    hf_repo: "microsoft/Phi-3.5-mini-instruct"
    quant: "4bit_nf4"
    max_new_tokens_classify: 128
    max_new_tokens_score: 24
    max_new_tokens_compress: 512
    temperature: 0.0

  - id: "qwen2.5-3b-instruct"
    hf_repo: "Qwen/Qwen2.5-3B-Instruct"
    quant: "4bit_nf4"
    max_new_tokens_classify: 128
    max_new_tokens_score: 24
    max_new_tokens_compress: 512
    temperature: 0.0

  - id: "qwen2.5-1.5b-instruct"
    hf_repo: "Qwen/Qwen2.5-1.5B-Instruct"
    quant: "4bit_nf4"
    max_new_tokens_classify: 128
    max_new_tokens_score: 24
    max_new_tokens_compress: 512
    temperature: 0.0

gates:
  stage1_advance:
    accuracy_overall: 0.55
    f1_macro: 0.50
    f1_hybrid_agentic: 0.40
    parse_rate: 0.95
    latency_mean_ms: 5000

  stage2_pipeline:
    routing_accuracy: 0.85
    compression_similarity: 0.88
    hypervisor_recall: 0.75
    hypervisor_precision: 0.80
    hypervisor_fpr: 0.10
    pipeline_p95_ms: 200

  stage4_rct_promotion:
    cost_reduction_pct: 25.0
    defect_rate_reduction_pct: 40.0
    quality_score_floor: 7.0

mlflow:
  tracking_uri: "http://localhost:5000"
  experiments:
    benchmark: "plexor/slm-benchmark-v2"
    integration: "plexor/slm-integration-v2"
    foundry_latency: "plexor/foundry-latency-v2"
    rct_prefix: "plexor/rct-arm"
    rct_analysis: "plexor/rct-analysis-v2"
```

### 7.3 RCT Config (`config/rct_config.yaml`)

```yaml
rct:
  version: "2.0"
  arms:
    A:
      label: "Control — auto"
      routing_enabled: false
      hypervisor_enabled: false
      frontend_model: null
      frontend_runtime: null
      backend: "anthropic_direct"

    B:
      label: "Phi-4-mini pretrained"
      routing_enabled: true
      hypervisor_enabled: true
      frontend_model: "phi-4-mini-instruct"
      frontend_runtime: "foundry"  # or "vllm" if Foundry latency fails (Section 3.5.5)
      frontend_endpoint: "${FOUNDRY_FRONTDOOR_URL}"  # or http://localhost:8000
      backend: "plexor_gateway_phi"
      vllm_config:  # used only if frontend_runtime == vllm
        model: "microsoft/Phi-4-mini-instruct"
        quantization: "bitsandbytes"
        max_model_len: 4096
        port: 8000

    C:
      label: "Qwen2.5-3B pretrained"
      routing_enabled: true
      hypervisor_enabled: true
      frontend_model: "qwen2.5-3b-instruct"
      frontend_runtime: "vllm"  # Qwen has no Foundry endpoint; always self-hosted
      frontend_endpoint: "http://localhost:8000"
      backend: "plexor_gateway_qwen"
      vllm_config:
        model: "Qwen/Qwen2.5-3B-Instruct"
        quantization: "bitsandbytes"
        max_model_len: 4096
        port: 8000

    D:
      label: "Hybrid split"
      routing_enabled: true
      hypervisor_enabled: true
      frontend_model: "hybrid_split"
      frontend_runtime: "mixed"  # Foundry for Phi families, vLLM for Qwen families
      split_policy:
        code_simple: "phi-4-mini-instruct"
        code_complex: "phi-4-mini-instruct"
        cot_simple: "phi-4-mini-instruct"
        cot_complex: "phi-4-mini-instruct"
        hybrid_agentic: "qwen2.5-3b-instruct"
        hybrid_generative: "phi-4-mini-instruct"
      split_endpoints:
        phi: "${FOUNDRY_FRONTDOOR_URL}"  # or http://localhost:8000
        qwen: "http://localhost:8001"     # second vLLM instance on different port
      backend: "plexor_gateway_hybrid"

  randomisation:
    method: "sha256_mod"
    n_arms: 4
    level: "session"

  sample_size:
    target_per_arm: 250
    total: 1000
    power: 0.80
    alpha: 0.05
    correction: "bonferroni"

  endpoints:
    provider_fast: "${PROVIDER_FAST_URL}"
    provider_mid: "${PROVIDER_MID_URL}"
    provider_frontier: "${PROVIDER_FRONTIER_URL}"

  logging:
    buffer_size: 10
    flush_interval_s: 30
```

### 7.4 Provider Tier Mapping (`config/provider_tiers.yaml`)

```yaml
tiers:
  fast:
    providers:
      - name: "anthropic"
        model: "claude-haiku-4-5"
        cost_per_1m_input: 0.80
        cost_per_1m_output: 4.00
      - name: "deepseek"
        model: "deepseek-chat"
        cost_per_1m_input: 0.27
        cost_per_1m_output: 1.10
      - name: "mistral"
        model: "mistral-small-latest"
        cost_per_1m_input: 0.20
        cost_per_1m_output: 0.60
    default: "deepseek"

  mid:
    providers:
      - name: "anthropic"
        model: "claude-sonnet-4-6"
        cost_per_1m_input: 3.00
        cost_per_1m_output: 15.00
      - name: "openai"
        model: "gpt-4o"
        cost_per_1m_input: 2.50
        cost_per_1m_output: 10.00
      - name: "google"
        model: "gemini-2.0-flash"
        cost_per_1m_input: 0.10
        cost_per_1m_output: 0.40
    default: "anthropic"

  frontier:
    providers:
      - name: "anthropic"
        model: "claude-opus-4-6"
        cost_per_1m_input: 15.00
        cost_per_1m_output: 75.00
      - name: "openai"
        model: "o3"
        cost_per_1m_input: 10.00
        cost_per_1m_output: 40.00
    default: "anthropic"
```

---

## 8. Decision Gates Summary

| Gate | Stage | Metric | Threshold | Blocks |
|---|---|---|---|---|
| G1: Model accuracy | 1 | accuracy_overall | ≥0.55 | Stage 2 entry |
| G2: Parse reliability | 1 | parse_rate | ≥0.95 | Stage 2 entry |
| G3: Latency ceiling | 1 | latency_mean_ms | ≤5,000 | Stage 2 entry |
| G4: Routing accuracy | 2 | routing_accuracy | ≥0.85 | Stage 4 entry |
| G5: Compression quality | 2 | semantic_similarity_avg | ≥0.88 | Stage 4 entry |
| G6: Hypervisor recall | 2 | hypervisor_recall | ≥0.75 | Stage 4 entry |
| G7: Hypervisor FPR | 2 | hypervisor_fpr | ≤0.10 | Stage 4 entry |
| G8: Foundry latency | 3 | pipeline_p95_ms | ≤200ms | Stage 4 entry |
| G9: Cost reduction | 4 | cost_reduction_pct | ≥25% | Production promotion |
| G10: Defect reduction | 4 | defect_rate_reduction_pct | ≥40% | Production promotion |
| G11: Quality floor | 4 | quality_score_avg | ≥7.0 | Production promotion |

Every production promotion requires an MLflow run ID where `gate_all_green = 1`. This run ID is recorded in the deployment PR and is mandatory for merge approval.

---

## 9. Timeline

| Week | Stage | Activities | Deliverable |
|---|---|---|---|
| 1 | Setup | VM provisioned, MLflow running, corpus expanded to 120 cases | Corpus JSONL + SHA256 logged |
| 1–2 | Stage 1 | Run all 4 models × 120 cases, significance testing | Benchmark report, 2 models advanced |
| 2–3 | Stage 2 | Full pipeline integration with advancing models | Pipeline validation report, gate check |
| 3 | Stage 3 | Foundry latency comparison | Latency report, deployment decision |
| 3–4 | Stage 4 | 4-arm RCT execution (250 per arm) | RCT analysis, promotion recommendation |
| 5+ | Stage 5 | Fine-tuning on winning model (if warranted) | Fine-tuned model, re-run gates |

Estimated total compute cost: $150–250 (VM spot hours + Foundry tokens + downstream provider calls).

---

## 10. Reproducibility

### 10.1 Immutable Artifacts

Every benchmark run produces:
- Corpus file hash (SHA256) logged to MLflow params
- Git commit hash logged to MLflow params
- VM hostname and GPU model logged to MLflow params
- Full per-case results logged as MLflow artifact
- Significance report logged as MLflow artifact

### 10.2 Reproducibility Script

```bash
# Export full reproducibility bundle
python scripts/export_repro_bundle.py \
    --experiment "plexor/slm-benchmark-v2" \
    --output repro/benchmark_v2_bundle/
```

Bundle contents: corpus JSONL, config YAML, per-case results JSON, significance report JSON, MLflow run metadata, git diff (if uncommitted changes), environment snapshot (`pip freeze`).

### 10.3 Artifact Repository

Frozen artifacts pushed to: `micoverde/plexor-rct-repro-data` (private)
Tag convention: `experiment-v2-{stage}-{date}`

---

## 11. Open Questions Carried Forward

| # | Question | Impact | Owner | Target |
|---|---|---|---|---|
| OQ-01 | Should hypervisor run on all requests or only code/complex + hybrid/agentic? | ~$0.008/request cost | ML Platform | Before Stage 4 |
| OQ-02 | Is Qwen2.5-3B a better hybrid_agentic candidate than Qwen2.5-1.5B? | Arm C model selection | Stage 1 resolves | Week 2 |
| OQ-04 | Is the 0.88 similarity threshold optimal for compression? | Compression acceptance rate | Stage 2 resolves | Week 3 |
| OQ-06 | Should Arm D's split policy be hard-coded or learned from Stage 1 results? | Arm D configuration | ML Platform | Before Stage 4 |
| OQ-07 | Can vLLM on local T4 match Foundry latency for Qwen (which has no Foundry endpoint)? | Arm C deployment | Stage 3 resolves | Week 3 |
| OQ-08 | Does vLLM `bitsandbytes` quantisation produce identical outputs to `transformers` `bitsandbytes`? | Accuracy continuity across runtimes | Runtime parity check (Section 3.5.6) | Before Stage 3 |
| OQ-09 | For Arm D hybrid split, can two vLLM instances run concurrently on T4 (Phi on port 8000, Qwen on 8001) with reduced `gpu-memory-utilization`? | Arm D feasibility | Stage 3 investigation | Week 3 |

---

## Document Control

| Field | Value |
|---|---|
| Version | 2.0 — Initial draft |
| Predecessor | Plexor_SLM_FrontDoor_Design_Spec_v1.0, RCT Article v1 |
| Next review | After Stage 1 benchmark results |
| Companion documents | MLFLOW_EXPERIMENT_DESIGN.md, PROGRESSIVE_TEST_CASES.md, EIR Acceptance Test v1.1 |
