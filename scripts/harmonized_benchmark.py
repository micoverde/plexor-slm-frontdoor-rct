#!/usr/bin/env python3
"""
Article 8 — Harmonized Offline Benchmark (Phase 1, Option C)

Runs all 3 evaluated models on the SAME corpus, SAME hardware, SAME serving
stack, SAME quantization — eliminating all cross-study confounds from v7.

Models (sequential, one at a time on the same T4):
    1. Phi-3.5-mini-instruct  (3.8B, 4-bit NF4)
    2. Qwen2.5-1.5B-Instruct  (1.5B, 4-bit NF4)
    3. Qwen2.5-3B-Instruct     (3.0B, 4-bit NF4)

Serving:  vLLM 0.17.1, bitsandbytes 4-bit, port 8002
Corpus:   progressive_test_cases_v2_60.jsonl  (SHA dac3aac5)
Contract: Frozen classification system prompt, max_new_tokens=128,
          temperature=0.0, greedy decoding

Usage (on Azure VM plexor-slm-bench-v2-westus2, inside ~/vllm-env):
    source ~/vllm-env/bin/activate
    python harmonized_benchmark.py \\
        --corpus ~/article8/progressive_test_cases_v2_60.jsonl \\
        --output-dir ~/article8/harmonized \\
        --port 8002

Output: 3 per-model JSON result files + cross-model significance report.
"""

import argparse
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Model configurations (locked)
# ---------------------------------------------------------------------------
MODELS = [
    {
        "name": "Phi-3.5-mini-instruct",
        "hf_id": "microsoft/Phi-3.5-mini-instruct",
        "shortname": "phi-3.5-mini",
        "params_b": 3.8,
    },
    {
        "name": "Qwen2.5-1.5B-Instruct",
        "hf_id": "Qwen/Qwen2.5-1.5B-Instruct",
        "shortname": "qwen2.5-1.5b",
        "params_b": 1.5,
    },
    {
        "name": "Qwen2.5-3B-Instruct",
        "hf_id": "Qwen/Qwen2.5-3B-Instruct",
        "shortname": "qwen2.5-3b",
        "params_b": 3.0,
    },
]

# ---------------------------------------------------------------------------
# Frozen prompts — identical to run_qwen3b_benchmark.py / benchmark_openweights.py
# ---------------------------------------------------------------------------
CLASSIFICATION_SYSTEM = """You are a task classifier for an AI routing system.
Classify the prompt into exactly one of these categories:
- code/simple       (single function, snippet, trivial script)
- code/complex      (multi-file, architecture-level, tests required)
- CoT/simple        (single-step explanation or reasoning)
- CoT/complex       (multi-step reasoning, trade-off analysis)
- hybrid/agentic    (autonomous execution, self-healing, multi-artifact, no-confirmation)
- hybrid/generative (mixed creative + structured output)

Key rule: autonomous mode + no confirmation + multi-artifact = hybrid/agentic.

Respond with JSON only: {"label": "<label>", "confidence": <0.0-1.0>}"""

SCORING_SYSTEM = """You are a complexity scorer for AI prompts.
Score from 1.0 to 10.0.
1-3: trivial function. 4-5: moderate multi-step. 6-7: complex multi-file.
8-10: autonomous loop, self-healing, TDD, git ops, multi-artifact.

Respond with JSON only: {"score": <float 1.0-10.0>, "reasoning": "<one sentence>"}"""

VALID_LABELS = {
    "code/simple", "code/complex",
    "CoT/simple", "CoT/complex",
    "hybrid/agentic", "hybrid/generative",
}

CLS_MAX_TOKENS = 128
TEMPERATURE = 0.0
VLLM_API_KEY = "local-vllm"
HEALTH_TIMEOUT_S = 300          # 5 min for model loading
HEALTH_POLL_INTERVAL_S = 5
REQUEST_TIMEOUT_S = 120
GPU_COOLDOWN_S = 10             # seconds between model swaps


# ---------------------------------------------------------------------------
# vLLM server lifecycle
# ---------------------------------------------------------------------------

def start_vllm_server(
    model_id: str,
    port: int,
    gpu_mem: float,
    max_model_len: int = 2048,
) -> subprocess.Popen:
    """Start vLLM OpenAI-compatible server as a subprocess."""
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_id,
        "--port", str(port),
        "--quantization", "bitsandbytes",
        "--load-format", "bitsandbytes",
        "--gpu-memory-utilization", str(gpu_mem),
        "--max-model-len", str(max_model_len),
        "--dtype", "float16",
        "--enforce-eager",           # avoid CUDA graph overhead on T4
        "--no-enable-log-requests",   # reduce log noise
        # Note: flashinfer JIT requires nvcc; install nvidia-cuda-nvcc-cu12 if missing
    ]
    print(f"  vLLM cmd: {' '.join(cmd)}")
    # Write vLLM output to a log file — do NOT use PIPE (causes deadlock
    # when the buffer fills up and the parent never reads it).
    log_dir = Path.home() / "article8" / "harmonized"
    log_dir.mkdir(parents=True, exist_ok=True)
    model_slug = model_id.replace("/", "_")
    vllm_log = open(log_dir / f"vllm_{model_slug}.log", "w")
    # Propagate critical env vars to child process:
    # - VLLM_USE_V1=0: use v0 engine (avoids v1 memory profiler assertion with co-resident GPU)
    # - VLLM_ATTENTION_BACKEND=FLASH_ATTN: avoid flashinfer JIT which needs nvcc
    env = os.environ.copy()
    env["VLLM_USE_V1"] = "0"
    env["VLLM_ATTENTION_BACKEND"] = "FLASH_ATTN"
    proc = subprocess.Popen(
        cmd,
        stdout=vllm_log,
        stderr=subprocess.STDOUT,
        env=env,
        start_new_session=True,  # create process group for clean kill
    )
    proc._vllm_log = vllm_log  # attach for cleanup
    return proc


def wait_for_health(base_url: str, timeout: int = HEALTH_TIMEOUT_S) -> bool:
    """Poll vLLM health AND verify models endpoint responds."""
    import requests as _req
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = _req.get(f"{base_url}/health", timeout=5)
            if r.status_code == 200:
                # Also verify the models endpoint works (engine core alive)
                m = _req.get(f"{base_url}/v1/models", timeout=10)
                if m.status_code == 200:
                    return True
                # Health OK but models not ready — engine still loading
        except Exception:
            pass
        time.sleep(HEALTH_POLL_INTERVAL_S)
    return False


def stop_vllm_server(proc: subprocess.Popen) -> None:
    """Kill vLLM process group (API server + engine cores) and free GPU."""
    if hasattr(proc, "_vllm_log"):
        proc._vllm_log.close()
    if proc.poll() is not None:
        return  # already dead
    # Kill entire process group — SIGTERM to group, then SIGKILL if needed
    pgid = os.getpgid(proc.pid)
    try:
        os.killpg(pgid, signal.SIGTERM)
        proc.wait(timeout=15)
    except (subprocess.TimeoutExpired, ProcessLookupError):
        try:
            os.killpg(pgid, signal.SIGKILL)
            proc.wait(timeout=10)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            pass
    # Also kill any remaining GPU processes on our port
    kill_port(8002)  # fallback
    # Give the GPU a moment to reclaim memory
    time.sleep(GPU_COOLDOWN_S)


def kill_port(port: int) -> None:
    """Kill any process listening on the given port."""
    result = subprocess.run(
        ["fuser", "-k", f"{port}/tcp"],
        capture_output=True,
    )
    if result.returncode == 0:
        print(f"  Killed existing process on port {port}")
        time.sleep(2)


def gpu_mem_free_mb() -> float:
    """Return free GPU memory in MB via nvidia-smi."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            text=True,
        )
        return float(out.strip().split("\n")[0])
    except Exception:
        return -1.0


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def call_vllm(
    base_url: str,
    model_name: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
) -> tuple:
    """Send a chat completion request. Returns (raw_text, latency_ms)."""
    import requests as _req
    t0 = time.perf_counter()
    resp = _req.post(
        f"{base_url}/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {VLLM_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": max_tokens,
            "temperature": TEMPERATURE,
        },
        timeout=REQUEST_TIMEOUT_S,
    )
    latency_ms = (time.perf_counter() - t0) * 1000
    resp.raise_for_status()
    text = resp.json()["choices"][0]["message"]["content"]
    return text, latency_ms


def parse_classification(raw: str) -> tuple:
    """Extract label and confidence. Returns (label, confidence, parse_ok)."""
    match = re.search(
        r'\{[^}]*"label"\s*:\s*"([^"]+)"[^}]*"confidence"\s*:\s*([\d.]+)[^}]*\}',
        raw,
    )
    if not match:
        # Try reversed key order
        match = re.search(
            r'\{[^}]*"confidence"\s*:\s*([\d.]+)[^}]*"label"\s*:\s*"([^"]+)"[^}]*\}',
            raw,
        )
        if match:
            return match.group(2), float(match.group(1)), True
        return "", 0.0, False
    return match.group(1), float(match.group(2)), True


def parse_score(raw: str) -> tuple:
    """Extract complexity score. Returns (score, parse_ok)."""
    match = re.search(r'"score"\s*:\s*([\d.]+)', raw)
    if not match:
        return 0.0, False
    return float(match.group(1)), True


# ---------------------------------------------------------------------------
# Single-model benchmark
# ---------------------------------------------------------------------------

def run_single_benchmark(
    base_url: str,
    model_config: dict,
    cases: list,
    corpus_sha: str,
) -> dict:
    """Run the full 60-case benchmark for one model. Returns result dict."""
    model_name = model_config["hf_id"]
    shortname = model_config["shortname"]
    total = len(cases)
    results = []
    correct = 0
    parsed = 0

    print(f"\n{'='*60}")
    print(f"  Benchmarking: {model_config['name']} ({model_config['params_b']}B)")
    print(f"  Cases: {total}  |  Endpoint: {base_url}")
    print(f"{'='*60}")

    for i, case in enumerate(cases):
        pid = case["id"]
        gt_label = case["gt_label"]
        prompt = case["prompt"]

        # --- Classification ---
        try:
            cls_raw, cls_latency = call_vllm(
                base_url, model_name, CLASSIFICATION_SYSTEM, prompt, CLS_MAX_TOKENS,
            )
            pred_label, pred_conf, cls_parse_ok = parse_classification(cls_raw)
        except Exception as e:
            cls_raw = str(e)
            cls_latency, pred_label, pred_conf, cls_parse_ok = 0.0, "", 0.0, False

        # --- Complexity scoring ---
        try:
            score_raw, score_latency = call_vllm(
                base_url, model_name, SCORING_SYSTEM, prompt, 128,
            )
            pred_score, score_parse_ok = parse_score(score_raw)
        except Exception as e:
            score_raw = str(e)
            score_latency, pred_score, score_parse_ok = 0.0, 0.0, False

        label_correct = pred_label == gt_label
        if label_correct:
            correct += 1
        if cls_parse_ok:
            parsed += 1

        score_in_range = (
            case.get("gt_score_min", 0) <= pred_score <= case.get("gt_score_max", 10)
        )

        result = {
            "prompt_id": pid,
            "model_shortname": shortname,
            "role": "frontdoor",
            "gt_label": gt_label,
            "predicted_label": pred_label,
            "label_correct": label_correct,
            "gt_score_min": case.get("gt_score_min", 0),
            "gt_score_max": case.get("gt_score_max", 10),
            "predicted_confidence": pred_conf,
            "predicted_score": pred_score,
            "score_in_range": score_in_range,
            "score_parse_success": score_parse_ok,
            "json_parse_success": cls_parse_ok,
            "latency_ms": cls_latency,
            "score_latency_ms": score_latency,
            "raw_output": cls_raw,
            "error": "",
        }
        results.append(result)

        status = "\u2713" if label_correct else "\u2717"
        print(
            f"  [{i+1:2d}/{total}] {pid} {status} "
            f"gt={gt_label:<20s} pred={pred_label:<20s} {cls_latency:7.0f}ms"
        )

    # --- Aggregate metrics ---
    accuracy = correct / total if total else 0
    parse_rate = parsed / total if total else 0
    latencies = [r["latency_ms"] for r in results if r["latency_ms"] > 0]
    avg_lat = sum(latencies) / len(latencies) if latencies else 0
    latencies_sorted = sorted(latencies)
    n_lat = len(latencies_sorted)
    median_lat = latencies_sorted[n_lat // 2] if n_lat else 0
    p95_lat = latencies_sorted[int(n_lat * 0.95)] if n_lat else 0
    p99_lat = latencies_sorted[int(n_lat * 0.99)] if n_lat else 0

    # Per-family breakdown
    fam_correct = defaultdict(int)
    fam_total = defaultdict(int)
    for r in results:
        fam = r["gt_label"]
        fam_total[fam] += 1
        if r["label_correct"]:
            fam_correct[fam] += 1

    family_accuracy = {
        fam: fam_correct[fam] / fam_total[fam]
        for fam in sorted(fam_total.keys())
    }

    print(f"\n  Accuracy:  {accuracy:.4f} ({correct}/{total})")
    print(f"  Parse:     {parse_rate:.4f} ({parsed}/{total})")
    print(f"  Latency:   mean={avg_lat:.0f}ms  median={median_lat:.0f}ms  p95={p95_lat:.0f}ms")
    for fam in sorted(fam_total.keys()):
        fc, ft = fam_correct[fam], fam_total[fam]
        print(f"    {fam:<20s}: {fc}/{ft} = {fc/ft:.2f}")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    return {
        "timestamp": timestamp,
        "benchmark_type": "harmonized_v1",
        "model": model_config["name"],
        "model_id": model_config["hf_id"],
        "model_shortname": shortname,
        "model_params_b": model_config["params_b"],
        "corpus_sha256": corpus_sha,
        "n_cases": total,
        "accuracy": accuracy,
        "parse_rate": parse_rate,
        "avg_latency_ms": round(avg_lat, 2),
        "median_latency_ms": round(median_lat, 2),
        "p95_latency_ms": round(p95_lat, 2),
        "p99_latency_ms": round(p99_lat, 2),
        "hardware": "Azure Standard_NC8as_T4_v3 (Tesla T4 16GB)",
        "vllm_version": "0.17.1",
        "quantization": "bitsandbytes 4-bit NF4",
        "cls_max_tokens": CLS_MAX_TOKENS,
        "temperature": TEMPERATURE,
        "gpu_shared": False,
        "gpu_shared_note": "Exclusive GPU — single model per run, no co-resident processes",
        "family_accuracy": family_accuracy,
        "results": results,
    }


# ---------------------------------------------------------------------------
# Cross-model statistics
# ---------------------------------------------------------------------------

def compute_mcnemar_exact(
    results_a: list,
    results_b: list,
    name_a: str,
    name_b: str,
) -> dict:
    """
    Exact McNemar test for paired classification on the same corpus.

    Constructs the 2x2 discordant table:
        b = cases where A correct, B wrong
        c = cases where A wrong,   B correct
    Two-sided p-value via binomial test on discordant pairs.
    """
    from scipy.stats import binomtest

    # Build lookup by prompt_id
    a_by_id = {r["prompt_id"]: r["label_correct"] for r in results_a}
    b_by_id = {r["prompt_id"]: r["label_correct"] for r in results_b}

    ids = sorted(set(a_by_id.keys()) & set(b_by_id.keys()))
    b_count = 0  # A correct, B wrong
    c_count = 0  # A wrong, B correct
    both_correct = 0
    both_wrong = 0

    for pid in ids:
        ac = a_by_id[pid]
        bc = b_by_id[pid]
        if ac and bc:
            both_correct += 1
        elif ac and not bc:
            b_count += 1
        elif not ac and bc:
            c_count += 1
        else:
            both_wrong += 1

    n_discordant = b_count + c_count
    if n_discordant == 0:
        p_value = 1.0
    else:
        result = binomtest(b_count, n_discordant, 0.5, alternative="two-sided")
        p_value = result.pvalue

    return {
        "model_a": name_a,
        "model_b": name_b,
        "n_paired": len(ids),
        "both_correct": both_correct,
        "a_correct_b_wrong": b_count,
        "a_wrong_b_correct": c_count,
        "both_wrong": both_wrong,
        "n_discordant": n_discordant,
        "p_value": round(p_value, 6),
        "significant_at_005": p_value < 0.05,
        "significant_at_001": p_value < 0.01,
    }


def compute_cross_model_report(all_results: dict, output_dir: Path) -> dict:
    """Compute all pairwise McNemar tests and save significance report."""
    models = sorted(all_results.keys())
    comparisons = []

    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            m_a, m_b = models[i], models[j]
            comp = compute_mcnemar_exact(
                all_results[m_a]["results"],
                all_results[m_b]["results"],
                m_a,
                m_b,
            )
            comparisons.append(comp)
            sig = "**" if comp["significant_at_001"] else ("*" if comp["significant_at_005"] else "ns")
            print(
                f"  McNemar {m_a} vs {m_b}: "
                f"b={comp['a_correct_b_wrong']} c={comp['a_wrong_b_correct']} "
                f"p={comp['p_value']:.4f} {sig}"
            )

    # Holm-Bonferroni correction
    sorted_comps = sorted(comparisons, key=lambda c: c["p_value"])
    k = len(sorted_comps)
    for rank, comp in enumerate(sorted_comps):
        adjusted_alpha = 0.05 / (k - rank)
        comp["holm_bonferroni_alpha"] = round(adjusted_alpha, 6)
        comp["holm_bonferroni_significant"] = comp["p_value"] < adjusted_alpha

    # Summary table
    summary = {
        "n_models": len(models),
        "models": {
            m: {
                "accuracy": all_results[m]["accuracy"],
                "parse_rate": all_results[m]["parse_rate"],
                "median_latency_ms": all_results[m]["median_latency_ms"],
                "p95_latency_ms": all_results[m]["p95_latency_ms"],
                "family_accuracy": all_results[m]["family_accuracy"],
            }
            for m in models
        },
        "corpus_sha256": next(iter(all_results.values()))["corpus_sha256"],
        "hardware": "Azure Standard_NC8as_T4_v3 (Tesla T4 16GB)",
        "quantization": "bitsandbytes 4-bit NF4",
        "vllm_version": "0.17.1",
    }

    report = {
        "generated": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "benchmark_type": "harmonized_v1",
        "summary": summary,
        "pairwise_mcnemar": comparisons,
    }

    report_path = output_dir / "harmonized_significance_report.json"
    # Convert numpy bools to Python bools for JSON serialization
    report_path.write_text(json.dumps(report, indent=2, default=lambda o: bool(o) if hasattr(o, '__bool__') and type(o).__name__ == 'bool_' else str(o)))
    print(f"\n  Significance report: {report_path}")
    return report


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def load_corpus(corpus_path: Path) -> tuple:
    """Load JSONL corpus and compute SHA-256. Returns (cases, sha_hex)."""
    if not corpus_path.exists():
        print(f"FATAL: Corpus not found at {corpus_path}")
        sys.exit(1)
    raw = corpus_path.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    cases = [json.loads(line) for line in corpus_path.read_text().strip().split("\n")]
    return cases, sha


def main():
    parser = argparse.ArgumentParser(
        description="Article 8 — Harmonized Offline Benchmark (Phase 1)",
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path.home() / "article8" / "progressive_test_cases_v2_60.jsonl",
        help="Path to locked JSONL corpus",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.home() / "article8" / "harmonized",
        help="Directory for result JSONs",
    )
    parser.add_argument("--port", type=int, default=8002, help="vLLM port (default 8002)")
    parser.add_argument("--gpu-mem", type=float, default=0.85, help="GPU memory utilization")
    parser.add_argument(
        "--models",
        nargs="*",
        choices=[m["shortname"] for m in MODELS],
        default=None,
        help="Run only specific models (default: all three)",
    )
    parser.add_argument(
        "--skip-vllm-management",
        action="store_true",
        help="Assume vLLM is already running; skip start/stop (for debugging)",
    )
    args = parser.parse_args()

    # Output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load corpus
    cases, corpus_sha = load_corpus(args.corpus)
    print(f"Corpus: {len(cases)} cases, SHA256: {corpus_sha[:16]}...")

    # Filter models if requested
    models_to_run = MODELS
    if args.models:
        models_to_run = [m for m in MODELS if m["shortname"] in args.models]

    base_url = f"http://127.0.0.1:{args.port}"
    all_results = {}

    print(f"\n{'#'*60}")
    print(f"  HARMONIZED BENCHMARK — {len(models_to_run)} models")
    print(f"  Corpus SHA: {corpus_sha[:16]}...")
    print(f"  Hardware:   Azure Standard_NC8as_T4_v3 (Tesla T4 16GB)")
    print(f"  Serving:    vLLM 0.17.1 + bitsandbytes 4-bit NF4")
    print(f"  Port:       {args.port}")
    print(f"{'#'*60}")

    for idx, model_config in enumerate(models_to_run):
        shortname = model_config["shortname"]
        print(f"\n[{idx+1}/{len(models_to_run)}] === {model_config['name']} ===")

        proc = None
        if not args.skip_vllm_management:
            # Kill anything on our port
            kill_port(args.port)

            # Check GPU memory
            free_mb = gpu_mem_free_mb()
            print(f"  GPU free: {free_mb:.0f} MB")

            # Start vLLM
            print(f"  Starting vLLM for {model_config['hf_id']}...")
            proc = start_vllm_server(
                model_config["hf_id"], args.port, args.gpu_mem,
            )

            # Wait for health
            print(f"  Waiting for vLLM health (up to {HEALTH_TIMEOUT_S}s)...")
            if not wait_for_health(base_url, HEALTH_TIMEOUT_S):
                print(f"  FATAL: vLLM failed to become healthy for {model_config['name']}")
                # Dump vLLM output for debugging
                if proc.poll() is not None:
                    stdout = proc.stdout.read() if proc.stdout else ""
                    print(f"  vLLM exited with code {proc.returncode}")
                    print(f"  Last output:\n{stdout[-2000:]}")
                stop_vllm_server(proc)
                continue
            print("  vLLM healthy!")

        # Run benchmark
        result = run_single_benchmark(base_url, model_config, cases, corpus_sha)
        all_results[shortname] = result

        # Save individual result
        out_path = args.output_dir / f"harmonized_{shortname}_{result['timestamp']}.json"
        out_path.write_text(json.dumps(result, indent=2))
        print(f"  Saved: {out_path}")

        # Stop vLLM
        if proc and not args.skip_vllm_management:
            print(f"  Stopping vLLM...")
            stop_vllm_server(proc)
            free_mb = gpu_mem_free_mb()
            print(f"  GPU free after shutdown: {free_mb:.0f} MB")

    # Cross-model statistics (need at least 2 models)
    if len(all_results) >= 2:
        print(f"\n{'#'*60}")
        print("  CROSS-MODEL SIGNIFICANCE ANALYSIS")
        print(f"{'#'*60}")
        try:
            report = compute_cross_model_report(all_results, args.output_dir)

            # Print summary table
            print(f"\n  {'Model':<20s} {'Accuracy':>10s} {'Parse':>8s} {'Med Lat':>10s} {'P95 Lat':>10s}")
            print(f"  {'-'*58}")
            for m in sorted(all_results.keys()):
                r = all_results[m]
                print(
                    f"  {m:<20s} {r['accuracy']:>10.4f} {r['parse_rate']:>8.4f} "
                    f"{r['median_latency_ms']:>10.0f}ms {r['p95_latency_ms']:>10.0f}ms"
                )
        except ImportError:
            print("  WARNING: scipy not available — skipping McNemar tests")
            print("  Install: pip install scipy")
            # Still save a partial report
            report = {
                "generated": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
                "benchmark_type": "harmonized_v1",
                "error": "scipy not available for McNemar tests",
                "models": {m: all_results[m]["accuracy"] for m in all_results},
            }
            report_path = args.output_dir / "harmonized_significance_report.json"
            report_path.write_text(json.dumps(report, indent=2))

    print(f"\n{'#'*60}")
    print(f"  HARMONIZED BENCHMARK COMPLETE")
    print(f"  Results in: {args.output_dir}")
    print(f"  Models completed: {len(all_results)}/{len(models_to_run)}")
    print(f"{'#'*60}")


if __name__ == "__main__":
    main()
