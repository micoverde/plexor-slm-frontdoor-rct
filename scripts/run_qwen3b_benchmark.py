#!/usr/bin/env python3
"""
Article 8 — Qwen-2.5-3B-Instruct classification benchmark.
Runs the frozen 60-case corpus against vLLM-served Qwen-2.5-3B
using the exact classification system prompt from benchmark_openweights.py.

Output: JSON results file with per-case accuracy, latency, parse rate.
"""
import json
import time
import sys
import hashlib
import requests
from datetime import datetime, timezone
from pathlib import Path

# === Configuration (locked) ===
VLLM_URL = "http://127.0.0.1:8001/v1/chat/completions"
VLLM_API_KEY = "local-vllm"
MODEL_NAME = "Qwen2.5-3B-Instruct"
CLS_MAX_TOKENS = 128
TEMPERATURE = 0.0
CORPUS_PATH = Path.home() / "article8" / "progressive_test_cases_v2_60.jsonl"

# Frozen classification system prompt (benchmark_openweights.py line 347)
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


def call_vllm(system_prompt: str, user_prompt: str, max_tokens: int) -> tuple[str, float]:
    """Send a chat completion request. Returns (raw_text, latency_ms)."""
    t0 = time.perf_counter()
    resp = requests.post(
        VLLM_URL,
        headers={"Authorization": f"Bearer {VLLM_API_KEY}", "Content-Type": "application/json"},
        json={
            "model": MODEL_NAME,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": max_tokens,
            "temperature": TEMPERATURE,
        },
        timeout=120,
    )
    latency_ms = (time.perf_counter() - t0) * 1000
    resp.raise_for_status()
    text = resp.json()["choices"][0]["message"]["content"]
    return text, latency_ms


def parse_classification(raw: str) -> tuple[str, float, bool]:
    """Extract label and confidence from raw model output. Returns (label, confidence, parse_ok)."""
    import re
    # Try to find JSON in the output
    match = re.search(r'\{[^}]*"label"\s*:\s*"([^"]+)"[^}]*"confidence"\s*:\s*([\d.]+)[^}]*\}', raw)
    if not match:
        # Try reversed key order
        match = re.search(r'\{[^}]*"confidence"\s*:\s*([\d.]+)[^}]*"label"\s*:\s*"([^"]+)"[^}]*\}', raw)
        if match:
            return match.group(2), float(match.group(1)), True
        return "", 0.0, False
    return match.group(1), float(match.group(2)), True


def parse_score(raw: str) -> tuple[float, bool]:
    """Extract score from raw model output. Returns (score, parse_ok)."""
    import re
    match = re.search(r'"score"\s*:\s*([\d.]+)', raw)
    if not match:
        return 0.0, False
    return float(match.group(1)), True


def main():
    # Load and hash corpus
    if not CORPUS_PATH.exists():
        print(f"ERROR: Corpus not found at {CORPUS_PATH}")
        sys.exit(1)

    corpus_bytes = CORPUS_PATH.read_bytes()
    corpus_sha = hashlib.sha256(corpus_bytes).hexdigest()
    cases = [json.loads(line) for line in CORPUS_PATH.read_text().strip().split("\n")]
    print(f"Corpus: {len(cases)} cases, SHA256: {corpus_sha[:16]}...")

    results = []
    correct = 0
    parsed = 0
    total = len(cases)

    for i, case in enumerate(cases):
        pid = case["id"]
        gt_label = case["gt_label"]
        prompt = case["prompt"]

        # Classification
        try:
            cls_raw, cls_latency = call_vllm(CLASSIFICATION_SYSTEM, prompt, CLS_MAX_TOKENS)
            pred_label, pred_conf, cls_parse_ok = parse_classification(cls_raw)
        except Exception as e:
            cls_raw, cls_latency, pred_label, pred_conf, cls_parse_ok = str(e), 0.0, "", 0.0, False

        # Scoring
        try:
            score_raw, score_latency = call_vllm(SCORING_SYSTEM, prompt, 128)
            pred_score, score_parse_ok = parse_score(score_raw)
        except Exception as e:
            score_raw, score_latency, pred_score, score_parse_ok = str(e), 0.0, 0.0, False

        label_correct = (pred_label == gt_label)
        if label_correct:
            correct += 1
        if cls_parse_ok:
            parsed += 1

        score_in_range = case.get("gt_score_min", 0) <= pred_score <= case.get("gt_score_max", 10)

        result = {
            "prompt_id": pid,
            "model_shortname": "qwen2.5-3b",
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

        status = "✓" if label_correct else "✗"
        print(f"  [{i+1:2d}/{total}] {pid} {status} gt={gt_label:<20s} pred={pred_label:<20s} {cls_latency:7.0f}ms")

    # Summary
    accuracy = correct / total if total else 0
    parse_rate = parsed / total if total else 0
    latencies = [r["latency_ms"] for r in results if r["latency_ms"] > 0]
    avg_lat = sum(latencies) / len(latencies) if latencies else 0
    latencies_sorted = sorted(latencies)
    median_lat = latencies_sorted[len(latencies_sorted) // 2] if latencies_sorted else 0
    p95_lat = latencies_sorted[int(len(latencies_sorted) * 0.95)] if latencies_sorted else 0

    print(f"\n{'='*60}")
    print(f"Model:     {MODEL_NAME}")
    print(f"Cases:     {total}")
    print(f"Accuracy:  {accuracy:.4f} ({correct}/{total})")
    print(f"Parse:     {parse_rate:.4f} ({parsed}/{total})")
    print(f"Latency:   mean={avg_lat:.0f}ms  median={median_lat:.0f}ms  p95={p95_lat:.0f}ms")
    print(f"{'='*60}")

    # Per-family breakdown
    from collections import defaultdict
    fam_correct = defaultdict(int)
    fam_total = defaultdict(int)
    for r in results:
        fam = r["gt_label"]
        fam_total[fam] += 1
        if r["label_correct"]:
            fam_correct[fam] += 1

    print("\nFamily breakdown:")
    for fam in sorted(fam_total.keys()):
        fc = fam_correct[fam]
        ft = fam_total[fam]
        print(f"  {fam:<20s}: {fc}/{ft} = {fc/ft:.2f}")

    # Write output
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = {
        "timestamp": timestamp,
        "model": MODEL_NAME,
        "model_id": "Qwen/Qwen2.5-3B-Instruct",
        "corpus_sha256": corpus_sha,
        "corpus_path": str(CORPUS_PATH),
        "n_cases": total,
        "accuracy": accuracy,
        "parse_rate": parse_rate,
        "avg_latency_ms": avg_lat,
        "median_latency_ms": median_lat,
        "p95_latency_ms": p95_lat,
        "hardware": "Azure Standard_NC8as_T4_v3 (Tesla T4 16GB)",
        "gpu_shared": True,
        "gpu_shared_note": "MLflow server co-resident; latency non-comparable to pilot",
        "vllm_version": "0.17.1",
        "quantization": "bitsandbytes 4bit",
        "cls_max_tokens": CLS_MAX_TOKENS,
        "temperature": TEMPERATURE,
        "family_accuracy": {fam: fam_correct[fam] / fam_total[fam] for fam in sorted(fam_total.keys())},
        "results": results,
    }

    out_path = Path.home() / "article8" / f"qwen3b_benchmark_{timestamp}.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()
