# Which Small Model Guards the Gate?

**Article 8 in the TAAC Research Series**

A locked benchmark and four-arm randomized controlled trial for production front-door LLM routing.

## Abstract

Production LLM inference systems increasingly deploy a *front-door router*—a small language model (SLM) that classifies incoming prompts by task type and complexity, then routes each request to an appropriately capable (and priced) downstream model. This paper reports two complementary studies for the Plexor front-door stack:

**Study 1 (Pilot Benchmark).** A locked, hardware-controlled pairwise comparison of Phi-3.5-mini and Qwen2.5-1.5B on a fixed 60-case corpus. Phi-3.5-mini achieves significantly higher exact-label accuracy (0.517 vs. 0.200; McNemar p = 0.0013), while Qwen2.5-1.5B is substantially faster (median latency 1,017 ms vs. 12,227 ms; Wilcoxon p = 1.63e-11). Family-level analysis reveals a split policy.

**Study 2 (Four-Arm RCT).** A pre-registered RCT comparing Phi-4-mini-instruct, Qwen-2.5-3B-Instruct, and DeepSeek-V3 against a no-routing control (N=250/arm, Bonferroni-corrected).

## Repository Structure

```
plexor-slm-frontdoor-rct/
├── paper/                      # LaTeX source and compiled PDF
│   ├── main.tex
│   ├── main.pdf
│   ├── references.bib
│   └── neurips_2024.sty
├── data/
│   ├── pilot_benchmark/        # Validated Study 1 results
│   │   ├── significance_report.json
│   │   ├── benchmark_config.json
│   │   ├── corpus_manifest.json
│   │   ├── family_decision_sheet.json
│   │   └── environment_manifest.json
│   └── rct_infrastructure/     # RCT validation data
├── scripts/                    # Experiment and analysis scripts
└── docs/                       # Protocol and structure docs
    ├── ARTICLE8_STRUCTURE.md
    └── EXPERIMENT_PROTOCOL.md
```

## Series Context

| Article | Title | Focus |
|---------|-------|-------|
| 1 | Compress or Route? | Task-dependent dichotomy |
| 2 | The Perplexity Paradox | Mechanistic explanation |
| 3 | Beyond the Cliff | Ultra-compression strategies |
| 4 | The Compression Paradox | Energy measurement |
| 5 | Compression Method Matters | Provider-dependent dynamics |
| 6 | Compression at Scale | Production RCT (N=358) |
| 7 | Beyond Natural Language | Novel prompt languages |
| **8** | **Which Small Model Guards the Gate?** | **SLM selection for routing** |

## Key Results (Study 1 — Validated)

| Metric | Phi-3.5-mini | Qwen2.5-1.5B | p-value |
|--------|-------------|-------------|---------|
| Exact accuracy | 0.517 | 0.200 | 0.0013 |
| JSON parse rate | 1.000 | 1.000 | 1.0 |
| Mean latency (ms) | 12,133 | 1,542 | 1.63e-11 |

**Split policy finding:** Phi dominates code/CoT families; Qwen dominates hybrid-agentic.

## Citation

```bibtex
@inproceedings{johnson2026gate,
  title={Which Small Model Guards the Gate? A Locked Benchmark and Four-Arm
         Randomized Controlled Trial for Production Front-Door Routing},
  author={Johnson, Warren},
  booktitle={arXiv preprint},
  year={2026},
  note={Article 8 in the TAAC Research Series}
}
```

## License

Research data and code released under MIT License. Paper content under CC-BY 4.0.
