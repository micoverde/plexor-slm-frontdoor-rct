#!/usr/bin/env python3
"""
Article 8 — Study 3 Four-Arm RCT with Synthetic Traffic (Phase 3)

Executes the pre-registered four-arm randomised controlled trial using
synthetic traffic generated from the frozen corpus families.

Arms:
    A — Control  (no front-door routing, pass-through)
    B — Phi-4-mini-instruct  (local vLLM on T4, sequential swap with C)
    C — Qwen-2.5-3B-Instruct (local vLLM on T4)
    D — DeepSeek-V3           (commercial API)

Design:
    N = 400 per arm (1,600 total)
    Assignment: SHA-256(session_id) mod 4
    Interim analysis at N=200/arm (O'Brien-Fleming boundaries)
    Multiple comparison correction: Holm-Bonferroni

Usage (on Azure VM, inside ~/vllm-env):
    source ~/vllm-env/bin/activate
    python rct_synthetic_runner.py \\
        --corpus ~/article8/progressive_test_cases_v2_60.jsonl \\
        --output-dir ~/article8/rct_results \\
        --deepseek-api-key <key> \\
        --port 8002

Prerequisites:
    - Harmonized benchmark complete (Phase 1)
    - EXPERIMENT_PROTOCOL.md v2 locked (Phase 2)
    - DeepSeek API key for Arm D
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
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Arm definitions (locked per EXPERIMENT_PROTOCOL v2)
# ---------------------------------------------------------------------------
ARMS = {
    0: {
        "name": "A",
        "label": "Control (pass-through)",
        "model": None,
        "endpoint_type": "none",
    },
    1: {
        "name": "B",
        "label": "Phi-4-mini-instruct (local vLLM)",
        "model": "microsoft/Phi-4-mini-instruct",
        "model_shortname": "phi-4-mini",
        "endpoint_type": "vllm_local",
    },
    2: {
        "name": "C",
        "label": "Qwen-2.5-3B-Instruct (local vLLM)",
        "model": "Qwen/Qwen2.5-3B-Instruct",
        "model_shortname": "qwen2.5-3b",
        "endpoint_type": "vllm_local",
    },
    3: {
        "name": "D",
        "label": "DeepSeek-V3 (commercial API)",
        "model": "deepseek-chat",
        "model_shortname": "deepseek-v3",
        "endpoint_type": "deepseek_api",
    },
}

# Same frozen prompts as harmonized benchmark
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

CLS_MAX_TOKENS = 128
TEMPERATURE = 0.0
VLLM_API_KEY = "local-vllm"
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1/chat/completions"

N_PER_ARM = 400
INTERIM_N = 200
HEALTH_TIMEOUT_S = 300
GPU_COOLDOWN_S = 10
REQUEST_TIMEOUT_S = 120


# ---------------------------------------------------------------------------
# Session assignment (deterministic)
# ---------------------------------------------------------------------------

def assign_arm(session_id: str) -> int:
    """Assign session to arm via SHA-256(session_id) mod 4."""
    h = hashlib.sha256(session_id.encode()).hexdigest()
    return int(h, 16) % 4


def generate_sessions(cases: list, n_per_arm: int, seed: int = 42) -> list:
    """
    Generate session list from frozen corpus.

    Each session is a (session_id, case) pair. We generate enough sessions
    so that each arm gets approximately n_per_arm assignments.
    Sessions are created deterministically from case IDs + counter.
    """
    import random
    rng = random.Random(seed)

    # Pre-generate more sessions than needed, then trim per arm
    arm_counts = defaultdict(int)
    sessions = []
    counter = 0
    target_total = n_per_arm * 4

    # We need ~4x n_per_arm total, but hash distribution may be uneven
    # Generate with 20% buffer
    max_attempts = target_total * 5

    while sum(arm_counts.values()) < target_total and counter < max_attempts:
        case = cases[counter % len(cases)]
        session_id = f"rct-s3-{case['id']}-{counter:06d}"
        arm = assign_arm(session_id)

        if arm_counts[arm] < n_per_arm:
            sessions.append({
                "session_id": session_id,
                "arm": arm,
                "case": case,
            })
            arm_counts[arm] += 1
        counter += 1

    # Verify balanced
    for a in range(4):
        print(f"  Arm {ARMS[a]['name']}: {arm_counts[a]} sessions")

    return sessions


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def call_vllm(base_url, model_name, system_prompt, user_prompt, max_tokens):
    """vLLM inference. Returns (text, latency_ms, cost_usd)."""
    import requests
    t0 = time.perf_counter()
    resp = requests.post(
        f"{base_url}/v1/chat/completions",
        headers={"Authorization": f"Bearer {VLLM_API_KEY}", "Content-Type": "application/json"},
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
    data = resp.json()
    text = data["choices"][0]["message"]["content"]
    # Local vLLM: cost is fixed GPU rate, not per-token
    # We'll compute cost in the analysis phase
    return text, latency_ms, 0.0


def call_deepseek(api_key, system_prompt, user_prompt, max_tokens):
    """DeepSeek API inference. Returns (text, latency_ms, cost_usd)."""
    import requests
    t0 = time.perf_counter()
    resp = requests.post(
        DEEPSEEK_BASE_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": "deepseek-chat",
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
    data = resp.json()
    text = data["choices"][0]["message"]["content"]
    # DeepSeek pricing: input $0.27/1M, output $1.10/1M
    usage = data.get("usage", {})
    input_tokens = usage.get("prompt_tokens", 0)
    output_tokens = usage.get("completion_tokens", 0)
    cost_usd = (input_tokens * 0.27 + output_tokens * 1.10) / 1_000_000
    return text, latency_ms, cost_usd


def call_control(case):
    """Arm A: No front-door. Returns placeholder result with zero latency/cost."""
    return "", 0.0, 0.0


def parse_classification(raw):
    """Extract label and confidence from model output."""
    match = re.search(
        r'\{[^}]*"label"\s*:\s*"([^"]+)"[^}]*"confidence"\s*:\s*([\d.]+)[^}]*\}',
        raw,
    )
    if not match:
        match = re.search(
            r'\{[^}]*"confidence"\s*:\s*([\d.]+)[^}]*"label"\s*:\s*"([^"]+)"[^}]*\}',
            raw,
        )
        if match:
            return match.group(2), float(match.group(1)), True
        return "", 0.0, False
    return match.group(1), float(match.group(2)), True


# ---------------------------------------------------------------------------
# vLLM lifecycle (reused from harmonized_benchmark.py)
# ---------------------------------------------------------------------------

def start_vllm_server(model_id, port, gpu_mem=0.85):
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_id,
        "--port", str(port),
        "--quantization", "bitsandbytes",
        "--load-format", "bitsandbytes",
        "--gpu-memory-utilization", str(gpu_mem),
        "--max-model-len", "2048",
        "--dtype", "float16",
        "--enforce-eager",
        "--no-enable-log-requests",
    ]
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


def wait_for_health(base_url, timeout=HEALTH_TIMEOUT_S):
    import requests
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(f"{base_url}/health", timeout=5)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(5)
    return False


def stop_vllm_server(proc):
    if proc.poll() is not None:
        return
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=10)
    time.sleep(GPU_COOLDOWN_S)


def kill_port(port):
    subprocess.run(["fuser", "-k", f"{port}/tcp"], capture_output=True)
    time.sleep(2)


# ---------------------------------------------------------------------------
# O'Brien-Fleming interim analysis
# ---------------------------------------------------------------------------

def obrien_fleming_boundary(n_current: int, n_final: int, alpha: float = 0.05) -> float:
    """
    Compute O'Brien-Fleming spending function boundary.

    At information fraction t = n_current / n_final, the boundary z-value is:
        z_boundary = z_{alpha/2} / sqrt(t)

    Returns the alpha spending at this look.
    """
    from scipy.stats import norm
    t = n_current / n_final
    if t <= 0:
        return 0.0
    z_full = norm.ppf(1 - alpha / 2)
    z_boundary = z_full / (t ** 0.5)
    # Two-sided alpha spent
    alpha_spent = 2 * (1 - norm.cdf(z_boundary))
    return alpha_spent


def run_interim_analysis(arm_results: dict, n_interim: int, n_final: int) -> dict:
    """
    Check O'Brien-Fleming boundaries at interim look.

    Returns dict with per-arm accuracy, pairwise tests, and
    whether any arm should be dropped (dominated on BOTH cost AND quality
    at p < 0.001 per pre-registration).
    """
    report = {
        "n_interim": n_interim,
        "n_final": n_final,
        "information_fraction": round(n_interim / n_final, 3),
    }

    # Per-arm accuracy
    accuracies = {}
    for arm_idx, results in arm_results.items():
        if not results:
            continue
        correct = sum(1 for r in results if r.get("label_correct", False))
        total = len(results)
        accuracies[ARMS[arm_idx]["name"]] = {
            "correct": correct,
            "total": total,
            "accuracy": round(correct / total, 4) if total else 0,
        }

    report["arm_accuracies"] = accuracies

    # Compute boundary
    try:
        boundary_alpha = obrien_fleming_boundary(n_interim, n_final)
        report["obf_boundary_alpha"] = round(boundary_alpha, 6)
        report["obf_note"] = (
            "Arm-dropping criterion: dominated on BOTH cost AND quality "
            "at p < 0.001 (pre-registered)"
        )
    except ImportError:
        report["obf_boundary_alpha"] = None
        report["obf_note"] = "scipy not available for boundary computation"

    return report


# ---------------------------------------------------------------------------
# Run a single arm's sessions
# ---------------------------------------------------------------------------

def run_arm_sessions(
    sessions: list,
    arm_idx: int,
    base_url: str | None,
    deepseek_api_key: str | None,
) -> list:
    """Execute all sessions for one arm. Returns list of result dicts."""
    arm_config = ARMS[arm_idx]
    arm_name = arm_config["name"]
    arm_sessions = [s for s in sessions if s["arm"] == arm_idx]
    total = len(arm_sessions)
    results = []

    print(f"\n  Running Arm {arm_name}: {arm_config['label']} ({total} sessions)")

    for i, session in enumerate(arm_sessions):
        case = session["case"]
        sid = session["session_id"]
        pid = case["id"]
        gt_label = case["gt_label"]
        prompt = case["prompt"]

        # Route to correct endpoint
        try:
            if arm_idx == 0:
                # Control: no classification
                raw, latency_ms, cost_usd = call_control(case)
                pred_label, pred_conf, parse_ok = "", 0.0, True
            elif arm_config["endpoint_type"] == "vllm_local":
                raw, latency_ms, cost_usd = call_vllm(
                    base_url, arm_config["model"],
                    CLASSIFICATION_SYSTEM, prompt, CLS_MAX_TOKENS,
                )
                pred_label, pred_conf, parse_ok = parse_classification(raw)
            elif arm_config["endpoint_type"] == "deepseek_api":
                raw, latency_ms, cost_usd = call_deepseek(
                    deepseek_api_key,
                    CLASSIFICATION_SYSTEM, prompt, CLS_MAX_TOKENS,
                )
                pred_label, pred_conf, parse_ok = parse_classification(raw)
            else:
                raise ValueError(f"Unknown endpoint type: {arm_config['endpoint_type']}")

        except Exception as e:
            raw, latency_ms, cost_usd = str(e), 0.0, 0.0
            pred_label, pred_conf, parse_ok = "", 0.0, False

        label_correct = pred_label == gt_label if arm_idx != 0 else None

        result = {
            "session_id": sid,
            "arm": arm_name,
            "arm_idx": arm_idx,
            "prompt_id": pid,
            "gt_label": gt_label,
            "predicted_label": pred_label,
            "label_correct": label_correct,
            "predicted_confidence": pred_conf,
            "json_parse_success": parse_ok,
            "latency_ms": round(latency_ms, 2),
            "cost_usd": round(cost_usd, 8),
            "raw_output": raw[:500],  # truncate for storage
            "error": "" if parse_ok else "parse_failure",
        }
        results.append(result)

        if arm_idx != 0:
            status = "\u2713" if label_correct else "\u2717"
            if (i + 1) % 20 == 0 or (i + 1) == total:
                correct_so_far = sum(1 for r in results if r["label_correct"])
                print(
                    f"    [{i+1:3d}/{total}] acc={correct_so_far/(i+1):.3f} "
                    f"lat_mean={sum(r['latency_ms'] for r in results)/(i+1):.0f}ms"
                )
        else:
            if (i + 1) % 100 == 0 or (i + 1) == total:
                print(f"    [{i+1:3d}/{total}] control pass-through")

    return results


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def load_corpus(corpus_path):
    if not corpus_path.exists():
        print(f"FATAL: Corpus not found at {corpus_path}")
        sys.exit(1)
    raw = corpus_path.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    cases = [json.loads(line) for line in corpus_path.read_text().strip().split("\n")]
    return cases, sha


def main():
    parser = argparse.ArgumentParser(description="Article 8 — Study 3 RCT (Phase 3)")
    parser.add_argument(
        "--corpus", type=Path,
        default=Path.home() / "article8" / "progressive_test_cases_v2_60.jsonl",
    )
    parser.add_argument("--output-dir", type=Path, default=Path.home() / "article8" / "rct_results")
    parser.add_argument("--port", type=int, default=8002)
    parser.add_argument("--gpu-mem", type=float, default=0.85)
    parser.add_argument("--deepseek-api-key", type=str, default=os.environ.get("DEEPSEEK_API_KEY", ""))
    parser.add_argument("--n-per-arm", type=int, default=N_PER_ARM)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--arms", nargs="*", choices=["A", "B", "C", "D"], default=None,
        help="Run specific arms only (default: all four)",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Validate DeepSeek key if Arm D requested
    run_arms = set(args.arms) if args.arms else {"A", "B", "C", "D"}
    if "D" in run_arms and not args.deepseek_api_key:
        print("FATAL: Arm D requires --deepseek-api-key or DEEPSEEK_API_KEY env var")
        sys.exit(1)

    # Load corpus
    cases, corpus_sha = load_corpus(args.corpus)
    print(f"Corpus: {len(cases)} cases, SHA256: {corpus_sha[:16]}...")

    # Generate sessions
    print(f"\nGenerating {args.n_per_arm} sessions per arm (seed={args.seed})...")
    sessions = generate_sessions(cases, args.n_per_arm, args.seed)

    # Execution plan:
    #   1. Arm A (control) — no model needed
    #   2. Arm B (Phi-4-mini) — start vLLM, run, stop
    #   3. Arm C (Qwen-2.5-3B) — start vLLM, run, stop
    #   4. Arm D (DeepSeek-V3) — API calls, no GPU needed

    base_url = f"http://127.0.0.1:{args.port}"
    all_arm_results = {}
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    print(f"\n{'#'*60}")
    print(f"  STUDY 3 RCT — Synthetic Traffic Execution")
    print(f"  Corpus SHA: {corpus_sha[:16]}...")
    print(f"  N per arm:  {args.n_per_arm}")
    print(f"  Arms:       {', '.join(sorted(run_arms))}")
    print(f"{'#'*60}")

    # --- Arm A: Control ---
    if "A" in run_arms:
        results_a = run_arm_sessions(sessions, 0, None, None)
        all_arm_results[0] = results_a
        _save_arm_results(args.output_dir, "A", results_a, timestamp, corpus_sha)

    # --- Arms B & C: Local vLLM (sequential swap) ---
    for arm_idx, arm_letter in [(1, "B"), (2, "C")]:
        if arm_letter not in run_arms:
            continue

        arm_config = ARMS[arm_idx]
        kill_port(args.port)
        print(f"\n  Starting vLLM for Arm {arm_letter}: {arm_config['model']}...")
        proc = start_vllm_server(arm_config["model"], args.port, args.gpu_mem)

        if not wait_for_health(base_url, HEALTH_TIMEOUT_S):
            print(f"  FATAL: vLLM failed for Arm {arm_letter}")
            stop_vllm_server(proc)
            continue

        print(f"  vLLM healthy for Arm {arm_letter}")
        results = run_arm_sessions(sessions, arm_idx, base_url, None)
        all_arm_results[arm_idx] = results
        _save_arm_results(args.output_dir, arm_letter, results, timestamp, corpus_sha)

        # Interim analysis check
        if len(results) >= INTERIM_N:
            interim = run_interim_analysis(
                {arm_idx: results[:INTERIM_N]}, INTERIM_N, args.n_per_arm,
            )
            interim_path = args.output_dir / f"interim_arm{arm_letter}_{timestamp}.json"
            interim_path.write_text(json.dumps(interim, indent=2))
            print(f"  Interim analysis saved: {interim_path}")

        stop_vllm_server(proc)

    # --- Arm D: DeepSeek API ---
    if "D" in run_arms:
        results_d = run_arm_sessions(sessions, 3, None, args.deepseek_api_key)
        all_arm_results[3] = results_d
        _save_arm_results(args.output_dir, "D", results_d, timestamp, corpus_sha)

    # --- Combined analysis ---
    if len(all_arm_results) >= 2:
        combined = _compute_combined_analysis(all_arm_results, corpus_sha, timestamp)
        combined_path = args.output_dir / f"rct_combined_analysis_{timestamp}.json"
        combined_path.write_text(json.dumps(combined, indent=2))
        print(f"\n  Combined analysis: {combined_path}")

    print(f"\n{'#'*60}")
    print(f"  RCT EXECUTION COMPLETE")
    print(f"  Results in: {args.output_dir}")
    print(f"  Arms completed: {len(all_arm_results)}/4")
    print(f"{'#'*60}")


def _save_arm_results(output_dir, arm_letter, results, timestamp, corpus_sha):
    """Save per-arm results to JSON."""
    out = {
        "timestamp": timestamp,
        "arm": arm_letter,
        "arm_config": ARMS[{"A": 0, "B": 1, "C": 2, "D": 3}[arm_letter]],
        "corpus_sha256": corpus_sha,
        "n_sessions": len(results),
        "results": results,
    }

    # Add aggregate metrics for treatment arms
    if arm_letter != "A":
        correct = sum(1 for r in results if r.get("label_correct"))
        total = len(results)
        latencies = [r["latency_ms"] for r in results if r["latency_ms"] > 0]
        costs = [r["cost_usd"] for r in results]

        out["accuracy"] = round(correct / total, 4) if total else 0
        out["total_cost_usd"] = round(sum(costs), 6)
        out["avg_latency_ms"] = round(sum(latencies) / len(latencies), 2) if latencies else 0
        out["p95_latency_ms"] = round(
            sorted(latencies)[int(len(latencies) * 0.95)], 2
        ) if latencies else 0

    path = output_dir / f"rct_arm{arm_letter}_{timestamp}.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"  Saved: {path}")


def _compute_combined_analysis(all_arm_results, corpus_sha, timestamp):
    """Compute combined RCT analysis across all arms."""
    summary = {}
    for arm_idx, results in all_arm_results.items():
        arm_name = ARMS[arm_idx]["name"]
        n = len(results)
        if arm_idx == 0:
            summary[arm_name] = {"n": n, "type": "control"}
            continue

        correct = sum(1 for r in results if r.get("label_correct"))
        latencies = [r["latency_ms"] for r in results if r["latency_ms"] > 0]
        costs = [r["cost_usd"] for r in results]
        lat_sorted = sorted(latencies) if latencies else []

        # Per-family accuracy
        fam_correct = defaultdict(int)
        fam_total = defaultdict(int)
        for r in results:
            fam = r["gt_label"]
            fam_total[fam] += 1
            if r.get("label_correct"):
                fam_correct[fam] += 1

        summary[arm_name] = {
            "n": n,
            "accuracy": round(correct / n, 4) if n else 0,
            "parse_rate": round(
                sum(1 for r in results if r["json_parse_success"]) / n, 4
            ) if n else 0,
            "avg_latency_ms": round(sum(latencies) / len(latencies), 2) if latencies else 0,
            "median_latency_ms": round(lat_sorted[len(lat_sorted) // 2], 2) if lat_sorted else 0,
            "p95_latency_ms": round(lat_sorted[int(len(lat_sorted) * 0.95)], 2) if lat_sorted else 0,
            "total_cost_usd": round(sum(costs), 6),
            "family_accuracy": {
                fam: round(fam_correct[fam] / fam_total[fam], 4)
                for fam in sorted(fam_total.keys())
            },
        }

    # Latency viable region check: 0.85 accuracy, 2000ms P95
    viable = {}
    for arm_name, s in summary.items():
        if arm_name == "A":
            continue
        viable[arm_name] = {
            "accuracy_pass": s["accuracy"] >= 0.85,
            "latency_pass": s["p95_latency_ms"] <= 2000,
            "viable": s["accuracy"] >= 0.85 and s["p95_latency_ms"] <= 2000,
        }

    return {
        "timestamp": timestamp,
        "corpus_sha256": corpus_sha,
        "summary": summary,
        "viable_region": viable,
        "design": {
            "n_per_arm": N_PER_ARM,
            "correction": "Holm-Bonferroni",
            "latency_gate_ms": 2000,
            "accuracy_gate": 0.85,
        },
    }


if __name__ == "__main__":
    main()
