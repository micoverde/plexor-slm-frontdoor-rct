# Article 8 — Pre-Registration Protocol

## Study Title
Which Small Model Guards the Gate? A Locked Benchmark and Four-Arm RCT for Production Front-Door Routing

## Pre-Registration Date
2026-03-14

## Principal Investigator
Warren Johnson (Bona Opera Studios)

---

## 1. Study Design Overview

### Study 1: Pilot Locked Benchmark (COMPLETED)
- **Design:** Paired pairwise comparison
- **Models:** Phi-3.5-mini vs Qwen2.5-1.5B (4-bit NF4)
- **N:** 60 cases (6 families × 10)
- **Hardware:** RunPod Secure A40
- **Status:** COMPLETE — results in Sections 2-3 of main.tex

### Study 2: Four-Arm RCT (PENDING)
- **Design:** Four-arm parallel RCT with stratified randomization
- **Arms:**
  - A: Control (no routing, pass-through)
  - B: Phi-4-mini-instruct (Azure AI Foundry serverless)
  - C: Qwen-2.5-3B-Instruct (self-hosted vLLM on A10 GPU)
  - D: DeepSeek-V3 (commercial API, 671B MoE)
- **N:** 250 per arm (1,000 total)
- **Infrastructure:** RunPod for offline benchmark validation; Azure for production RCT

---

## 2. Hypotheses (Pre-Registered)

### H1: Routing Accuracy
- **Metric:** % requests routed to ground-truth tier
- **Expected:** D > C ≥ B > A
- **Threshold:** ≥85% for winning arm
- **Test:** Chi-squared, Bonferroni-corrected (α/6 = 0.0083)

### H2: Compression Quality
- **Metric:** Avg compression ratio at semantic similarity ≥0.88
- **Expected:** D > C ≥ B > A
- **Threshold:** ≥30% token reduction
- **Test:** Welch's t-test, Bonferroni-corrected

### H3: Total Cost
- **Metric:** USD per request (front-door + downstream)
- **Expected:** B or C < D < A
- **Threshold:** ≥25% reduction vs Arm A
- **Test:** Welch's t-test, Bonferroni-corrected

### H4: Defect Rate
- **Metric:** % responses with ≥1 hypervisor watch-point flag
- **Expected:** B/C/D < A
- **Threshold:** ≥40% reduction vs Arm A
- **Test:** Chi-squared, Bonferroni-corrected

### H5: Classification F1
- **Metric:** F1 macro; F1 on hybrid/agentic specifically
- **Expected:** D > C ≥ B
- **Threshold:** F1 macro ≥0.82
- **Test:** Chi-squared, Bonferroni-corrected

### H6: Latency SLA
- **Metric:** P95 front-door latency
- **Expected:** C ≤ B < D (network penalty)
- **Hard gate:** ≤200ms all arms
- **Test:** Bootstrap CI comparison

---

## 3. Statistical Analysis Plan

### Primary Analysis
- Conducted after all arms reach N=250
- Six pairwise comparisons per metric
- Bonferroni correction: α/6 = 0.0083
- Effect sizes: Cohen's h (proportions), Cohen's d (continuous) with 95% CIs

### Subgroup Analysis
- Per task-label analysis (exploratory, not Bonferroni-corrected)
- Flagged if p < 0.01
- Expected: largest divergence on hybrid/agentic and hybrid/generative

### Pareto Frontier
- Computed over (total_cost_per_request, routing_accuracy)
- Dominated arms identified
- Marginal cost per accuracy percentage point

### Interim Analysis
- At N=125 per arm
- O'Brien-Fleming spending function boundaries
- Arm-dropping criteria: dominated on BOTH cost AND quality at p < 0.001

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

## 5. Pilot-to-RCT Bridge

The pilot used **different model generations** than the RCT:

| Pilot Model | RCT Model | Change |
|-------------|-----------|--------|
| Phi-3.5-mini (3.8B) | Phi-4-mini-instruct (3.8B) | Same params, improved instruction following |
| Qwen2.5-1.5B | Qwen-2.5-3B-Instruct | 2× parameters |
| N/A | DeepSeek-V3 (671B MoE) | New arm |

**Transferability assessment:**
- Phi accuracy advantage expected to persist or improve
- Qwen latency advantage may narrow with 2× params
- DeepSeek: Articles 4-5 documented instability under compression; output budget (128 tokens) may mitigate for classification
- Pilot serves as **motivating evidence**, not combinable data

---

## 6. Cost Accounting (Locked at Trial Start)

| Arm | Cost Model | Rate |
|-----|-----------|------|
| A (Control) | $0 front-door | Downstream at auto rates |
| B (Phi) | Per-token × 3 calls | Input $0.10/1M, Output $0.10/1M |
| C (Qwen) | Fixed GPU / requests_per_hour | ~$0.60/hr per A10 replica |
| D (DeepSeek) | Per-token × 3 calls | Input $0.27/1M, Output $1.10/1M |

---

## 7. Infrastructure

### RunPod (Offline Benchmark Extension)
- GPU: NVIDIA A40 or A100
- Purpose: Extended offline benchmark, model comparison, failure analysis
- Account: Warren's RunPod account

### Azure (Production RCT)
- Arm B: Azure AI Foundry serverless (2 endpoints)
- Arm C: Container Apps GPU (1-3 A10 replicas, vLLM)
- Arm D: DeepSeek commercial API
- MLflow: Azure Container Apps + PostgreSQL + Blob Storage

---

## 8. Reproducibility Commitments

- All run IDs logged to MLflow with experiment tags
- Corpus SHA-256 hash sealed before arm activation
- Ground-truth labels sealed (κ ≥ 0.75 inter-annotator agreement required)
- Pricing snapshot artifact in MLflow (immutable)
- Source code and scripts in `scripts/ab-test/`
- Reproducibility bundle: `scripts/ab-test/export_repro_bundle.sh`

---

## 9. Ethical Disclosure

- Uses AI assistance (Claude) throughout for writing, code generation, and analysis
- Author affiliated with Bona Opera Studios (commercial interest in Plexor platform)
- Mitigation: pre-registration, public arXiv deposition, transparent reporting
- All findings reported regardless of direction (negative results included)
- No selective reporting; all six hypotheses reported

---

## 10. Timeline

| Phase | Target | Activities |
|-------|--------|-----------|
| Pre-registration | 2026-03-14 | This document sealed |
| Infrastructure | Week 1 | Provision Arm C GPU, configure Arm D, extend A/B to 4-arm |
| Corpus prep | Week 2 | Ground-truth labeling (1,000 prompts, 2 annotators, κ ≥ 0.75) |
| Trial run | Week 3 | 4-arm traffic split, N=250/arm, interim at N=125 |
| Analysis | Week 4 | rct_analysis.py, Pareto frontier, decision matrix |
| Paper completion | Week 5 | Fill pending sections, generate figures, final review |
| arXiv submission | Week 6 | Submit preprint |
