# Article 8 — Pre-Registration Protocol (v2)

## Study Title
Which Small Model Guards the Gate? A Locked Benchmark and Four-Arm RCT for Production Front-Door Routing

## Pre-Registration Date
2026-03-14 (v1), **2026-03-15 (v2 — Option C harmonized redesign)**

## Principal Investigator
Warren Johnson (Bona Opera Studios)

## v2 Change Log
- Replaced Studies 1 & 2 with harmonized three-model benchmark (same corpus/hardware/stack)
- Increased RCT sample size: N=400/arm (was 250), ~80% power for 10pp MDE
- Corrected multiple comparison procedure: Holm-Bonferroni primary (was plain Bonferroni)
- Updated latency gate: P95 ≤ 2,000ms (was 200ms — original was unrealistic for vLLM cold inference)
- Arm B: local vLLM Phi-4-mini on T4 (was Azure AI Foundry serverless — no Foundry access)
- Arm C: self-hosted vLLM on T4 (was A10 — matches actual infrastructure)
- Viable region: ≥0.85 accuracy AND ≤2,000ms P95 latency
- Interim analysis at N=200/arm (was N=125)
- Traffic source: synthetic corpus (was "live randomised controlled trial")

---

## 1. Study Design Overview

### Study 1: Harmonized Offline Benchmark (REPLACES v1 Studies 1 & 2)
- **Design:** Three-model sequential benchmark on identical infrastructure
- **Models:**
  1. Phi-3.5-mini-instruct (3.8B, 4-bit NF4)
  2. Qwen2.5-1.5B-Instruct (1.5B, 4-bit NF4)
  3. Qwen2.5-3B-Instruct (3.0B, 4-bit NF4)
- **N:** 60 cases (6 families × 10) — locked corpus SHA `dac3aac5`
- **Hardware:** Azure Standard_NC8as_T4_v3 (Tesla T4 16GB) — exclusive GPU per model
- **Serving:** vLLM 0.17.1, bitsandbytes 4-bit NF4, port 8002
- **Contract:** Frozen classification system prompt, max_new_tokens=128, temperature=0.0
- **Statistics:** Exact McNemar test (paired, same corpus), Holm-Bonferroni correction
- **Status:** IN PROGRESS — `scripts/harmonized_benchmark.py`

### Study 2: Four-Arm RCT (REDESIGNED)
- **Design:** Four-arm parallel RCT with deterministic randomization
- **Arms:**
  - A: Control (no routing, pass-through)
  - B: Phi-4-mini-instruct (local vLLM on T4, sequential swap with C)
  - C: Qwen-2.5-3B-Instruct (local vLLM on T4)
  - D: DeepSeek-V3 (commercial API, 671B MoE)
- **N:** 400 per arm (1,600 total) — **~80% power for 10pp minimum detectable effect**
- **Assignment:** SHA-256(session_id) mod 4
- **Traffic:** Synthetic corpus generated from frozen corpus families (SHA `dac3aac5`)
- **Infrastructure:** Azure Standard_NC8as_T4_v3 for Arms B/C; DeepSeek API for Arm D
- **Status:** PENDING — blocked on Phase 1 completion

---

## 2. Hypotheses (Pre-Registered)

### H1: Routing Accuracy
- **Metric:** % requests routed to ground-truth tier
- **Expected:** D > C ≥ B > A
- **Threshold:** ≥85% for winning arm
- **Test:** Chi-squared, Holm-Bonferroni-corrected

### H2: Compression Quality
- **Metric:** Avg compression ratio at semantic similarity ≥0.88
- **Expected:** D > C ≥ B > A
- **Threshold:** ≥30% token reduction
- **Test:** Welch's t-test, Holm-Bonferroni-corrected

### H3: Total Cost
- **Metric:** USD per request (front-door + downstream)
- **Expected:** B or C < D < A
- **Threshold:** ≥25% reduction vs Arm A
- **Test:** Welch's t-test, Holm-Bonferroni-corrected

### H4: Defect Rate
- **Metric:** % responses with ≥1 hypervisor watch-point flag
- **Expected:** B/C/D < A
- **Threshold:** ≥40% reduction vs Arm A
- **Test:** Chi-squared, Holm-Bonferroni-corrected

### H5: Classification F1
- **Metric:** F1 macro; F1 on hybrid/agentic specifically
- **Expected:** D > C ≥ B
- **Threshold:** F1 macro ≥0.82
- **Test:** Chi-squared, Holm-Bonferroni-corrected

### H6: Latency SLA
- **Metric:** P95 front-door latency
- **Expected:** C ≤ B < D (network penalty)
- **Hard gate:** ≤2,000ms P95 all arms
- **Test:** Bootstrap CI comparison

---

## 3. Statistical Analysis Plan

### Primary Analysis
- Conducted after all arms reach N=400
- Six hypotheses with Holm-Bonferroni step-down correction (primary)
  - Rank p-values p_(1) ≤ ... ≤ p_(6)
  - Reject p_(i) if p_(i) < α / (6 - i + 1), stop at first non-rejection
- Effect sizes: Cohen's h (proportions), Cohen's d (continuous) with 95% CIs

### Power Justification (v2)
- N=400/arm provides ~80% power for a 10 percentage-point difference in accuracy
  (two-proportion z-test, α=0.05/6 Bonferroni-adjusted)
- v1 N=250 provided only ~61% power — insufficient for reliable detection

### Subgroup Analysis
- Per task-label analysis (exploratory, not corrected)
- Flagged if p < 0.01
- Expected: largest divergence on hybrid/agentic and hybrid/generative

### Pareto Frontier
- Computed over (total_cost_per_request, routing_accuracy)
- Dominated arms identified
- Marginal cost per accuracy percentage point

### Interim Analysis
- At N=200 per arm (information fraction t=0.5)
- O'Brien-Fleming spending function boundaries
- Arm-dropping criteria: dominated on BOTH cost AND quality at p < 0.001
- If no arm dropped, continue to N=400

### Viable Region (Pre-Registered)
An arm is **viable** if it meets BOTH:
1. Classification accuracy ≥ 0.85
2. P95 front-door latency ≤ 2,000ms

Non-viable arms are excluded from the decision matrix regardless of other metrics.

### Failure Mode Coding
- Random sample of 50 misclassified cases per arm
- Four-dimension coding:
  1. Label confusion type (adjacent-family vs random)
  2. Confidence calibration (high-confidence errors vs low)
  3. Prompt characteristics (length, ambiguity, domain vocabulary)
  4. Output pathology (truncation, hallucinated labels, JSON failures)

---

## 4. Decision Matrix (Pre-Registered)

| Scenario | Cost Winner | Quality Winner | Decision |
|----------|-------------|----------------|----------|
| SLM wins both | B or C | B or C | Deploy winning SLM |
| DeepSeek quality, SLM cost | B or C | D | SLM + DeepSeek fallback for hard labels |
| DeepSeek wins both | D | D | DeepSeek primary; SLM fallback |
| All similar quality | Lowest cost | Comparable | Deploy cheapest |
| DeepSeek fails latency | Any | Any | Eliminate D; choose B vs C |

**Tie-breaking:** Prefer fewer external dependencies (self-hosted > serverless > API)

---

## 5. Harmonized Benchmark Bridge (v2)

### Why Harmonized? (Addresses v7 Confounds)
v1 Studies 1 & 2 had cross-study confounds:
- **Different hardware:** Study 1 on RunPod A40, Study 2 on Azure T4
- **Different serving stacks:** Study 1 via transformers, Study 2 via vLLM
- **Different corpus SHAs:** Corpus reconstructed between studies
- **GPU sharing:** Study 2 had MLflow co-resident on GPU

The harmonized benchmark eliminates ALL confounds by running all 3 models
sequentially on identical infrastructure with exclusive GPU access.

### Harmonized → RCT Bridge
| Harmonized Model | RCT Model | Change |
|------------------|-----------|--------|
| Phi-3.5-mini (3.8B) | Phi-4-mini-instruct (3.8B) | Same params, improved instruction following |
| Qwen2.5-1.5B | — | Baseline only (not in RCT) |
| Qwen2.5-3B | Qwen-2.5-3B-Instruct | Same model |
| — | DeepSeek-V3 (671B MoE) | New arm |

**Transferability assessment:**
- Phi accuracy advantage expected to persist or improve with Phi-4-mini
- Qwen-2.5-3B: directly carried from harmonized to RCT (same model, same hardware)
- DeepSeek: 671B MoE; Articles 4-5 documented instability under compression but output budget (128 tokens) may mitigate for classification
- Harmonized benchmark serves as **calibration data**, not combinable with RCT

---

## 6. Cost Accounting (Locked at Trial Start)

| Arm | Cost Model | Rate |
|-----|-----------|------|
| A (Control) | $0 front-door | Downstream at auto rates |
| B (Phi-4-mini) | Fixed GPU / requests_per_hour | ~$0.75/hr per T4 (on-demand) |
| C (Qwen-3B) | Fixed GPU / requests_per_hour | ~$0.75/hr per T4 (on-demand) |
| D (DeepSeek) | Per-token × 3 calls | Input $0.27/1M, Output $1.10/1M |

**Note (v2):** Arms B and C share the same T4 GPU via sequential model swap.
Cost is amortized GPU-hour rate divided by requests served during that period.

---

## 7. Infrastructure

### Azure VM (Harmonized Benchmark + RCT Arms B/C)
- **VM:** `plexor-slm-bench-v2-westus2` (Standard_NC8as_T4_v3)
- **IP:** 4.155.192.61
- **GPU:** NVIDIA Tesla T4 16GB
- **Serving:** vLLM 0.17.1 in `~/vllm-env`, bitsandbytes 4-bit NF4
- **Port:** 8002 (avoids conflict with plexor user's server on 8001)
- **Cost:** ~$0.75/hr on-demand, ~$0.35/hr spot

### DeepSeek API (RCT Arm D)
- **Endpoint:** `https://api.deepseek.com/v1/chat/completions`
- **Model:** `deepseek-chat` (DeepSeek-V3, 671B MoE)
- **Authentication:** API key (stored in env var `DEEPSEEK_API_KEY`)

### MLflow (Experiment Tracking)
- **Location:** Azure Container Apps + PostgreSQL + Blob Storage
- **Purpose:** All run IDs, artifacts, and metrics logged per arm

---

## 8. Reproducibility Commitments

- All run IDs logged to MLflow with experiment tags
- Corpus SHA-256 hash sealed before arm activation: `dac3aac5`
- Ground-truth labels sealed (κ ≥ 0.75 inter-annotator agreement required)
- Pricing snapshot artifact in MLflow (immutable)
- Source code: `scripts/harmonized_benchmark.py`, `scripts/rct_synthetic_runner.py`
- Reproducibility bundle: `scripts/ab-test/export_repro_bundle.sh`
- Harmonized benchmark: all 3 models produce results with identical corpus SHA, hardware string, and vLLM version

---

## 9. Ethical Disclosure

- Uses AI assistance (Claude) throughout for writing, code generation, and analysis
- Author affiliated with Bona Opera Studios (commercial interest in Plexor platform)
- Mitigation: pre-registration, public arXiv deposition, transparent reporting
- All findings reported regardless of direction (negative results included)
- No selective reporting; all six hypotheses reported
- v2 protocol update motivated by peer review feedback on v1 confounds

---

## 10. Timeline

| Phase | Target | Activities |
|-------|--------|-----------|
| Pre-registration v1 | 2026-03-14 | Original document sealed |
| **Pre-registration v2** | **2026-03-15** | **Option C redesign: harmonized benchmark + RCT power increase** |
| Harmonized benchmark | 2026-03-15 | Run 3 models sequentially on Azure T4 (~2 hours) |
| RCT design lock | 2026-03-16 | Verify protocol v2, lock corpus SHA for RCT |
| RCT execution | 2026-03-17–18 | 4-arm synthetic traffic, N=400/arm, interim at N=200 |
| Analysis | 2026-03-19 | Holm-Bonferroni tests, Pareto frontier, decision matrix |
| Paper v8 | 2026-03-20–21 | Replace Studies 1&2, fill Study 3, compile PDF |
| arXiv submission | 2026-03-22 | Submit preprint |
