# Article 8: Which Small Model Guards the Gate?

## Status: SCAFFOLD COMPLETE — RCT EXECUTION PENDING

**Article 8 in the TAAC Research Series**
**Target venue:** arXiv preprint (NeurIPS format)
**Date:** March 2026

---

## Validated vs. Pending

### Validated (publishable today as pilot report)
- [x] Locked 60-case pairwise benchmark (Phi-3.5-mini vs Qwen2.5-1.5B)
- [x] Statistical significance on primary endpoint (McNemar p=0.0013)
- [x] Family-level split policy finding
- [x] RCT infrastructure validation (preflight, progressive-15, bulk-250)
- [x] Pre-warm effect (MBPP case study)
- [x] Reproducibility manifest (run IDs, SHA hashes, artifacts)

### Pending (requires RCT execution)
- [ ] Four-arm RCT with endpoint-isolated arms
- [ ] Phi-4-mini vs Qwen-2.5-3B vs DeepSeek-V3 vs Control
- [ ] N≥250 per arm with Bonferroni correction
- [ ] Pareto frontier (cost, accuracy) visualization
- [ ] Pre-registered decision matrix application
- [ ] Failure mode coding (qualitative error analysis)
- [ ] Production deployment recommendation

---

## Paper Structure

### Section 1: Introduction (routing as model selection)
- Front-door routing as model-selection problem
- Connection to Article 1 (task dichotomy → now extended to model selection)
- Three contributions: pilot benchmark + RCT design + RCT results

### Section 2: Study 1 — Pilot Locked Benchmark
- Source: Current PDF (Plexor_SLM_FrontDoor_RCT_Article_v1.pdf)
- Experimental contract, endpoints, results
- Family-level behavior and split policy finding
- **NEW: Pilot-to-RCT bridge** (model generation change 3.5→4, 1.5B→3B)

### Section 3: Study 2 — Four-Arm RCT Design
- Source: Condensed from Plexor_SLM_Selection_RCT_Design_v1_0.tex
- Arms, hypotheses, randomization, power, statistical analysis plan
- Pre-registered decision matrix
- Infrastructure validation results

### Section 4: RCT Infrastructure Validation
- Preflight, progressive ingestion, bulk logging
- Pre-warm effect
- Endpoint aliasing caveat

### Section 5: Study 2 — RCT Results [PENDING]
- Primary outcomes per arm
- Pairwise comparisons (Bonferroni)
- Family-level subgroup analysis
- **Failure mode coding** (linguistic error analysis)
- Interim analysis

### Section 6: Pareto Analysis and Decision [PENDING]
- Cost-quality frontier
- Decision matrix application
- Production deployment recommendation

### Section 7: Discussion
- Routing as task-conditional model selection
- Connection to Articles 4-5 (DeepSeek provider dynamics)
- Methodological advancement over Article 6
- Limitations

### Section 8: Threats to Validity
- Internal, external, construct validity

### Section 9: Reproducibility
- Study 1 and Study 2 artifacts
- Run IDs, scripts, automation

### Section 10: Conclusion

---

## Series Connections Map

```
Art 1 (Compress or Route)
  └── Task dichotomy: compress code, route CoT
       └── Art 8: "Which model should DO the routing?"

Art 4 (Green AI) + Art 5 (Token Explosion)
  └── DeepSeek 38× output explosion
       └── Art 8 Arm D: DeepSeek-V3 as classifier — explosion risk?

Art 6 (Compression RCT)
  └── CONSORT methodology, N=358, 6-arm
       └── Art 8: Same methodology, 4-arm, Bonferroni, Pareto

Art 7 (Prompt Languages)
  └── Tokenizer-awareness principle
       └── Art 8: Classifier prompts must be tokenizer-aware per SLM
```

---

## Key Differentiation from Prior Articles

| Dimension | Art 6 | Art 8 |
|-----------|-------|-------|
| Variable | Compression strategy | Model backbone |
| Arms | 6 (compression levels) | 4 (model backends) |
| Correction | Uncorrected exploratory | Bonferroni α/6=0.0083 |
| Analysis | Single-metric comparison | Pareto frontier + decision matrix |
| Error analysis | Accuracy only | Failure mode coding (qualitative) |
| Deployment | Threshold recommendation | Concrete arm selection |
| Cost model | Uniform (same model) | Heterogeneous (serverless, GPU, API) |

---

## Files in this directory

```
paper-v8/
├── main.tex                    # Full article LaTeX (scaffold)
├── references.bib              # Bibliography
├── neurips_2024.sty            # NeurIPS style file
├── ARTICLE8_STRUCTURE.md       # This file
├── EXPERIMENT_PROTOCOL.md      # Pre-registration document
├── data/                       # RCT data (after execution)
├── analysis/                   # Statistical analysis outputs
├── scripts/                    # Experiment and analysis scripts
├── figures/                    # Generated figures
└── docs/                       # Supporting documentation
```
