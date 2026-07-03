# Context Volume, Not Coupling: Constraint Recovery in Small Language-Model Loops

A minimal, fully reproducible perturbation–recovery experiment testing whether inter-agent coupling improves recovery from instruction perturbation in recursive LLM loops, or whether the effect is fully explained by per-turn generation volume.

## Core Result
In a paired design with 30 runs across 4 conditions on Qwen2.5-1.5B-Instruct:
- Coupled dual-model loops recover from perturbation more often than single self-continued loops (43% vs 17%).
- When a single model generates twice per turn — matching the coupled condition's total generation volume — recovery rate is statistically indistinguishable from the coupled loop (37% vs 43%, p=0.774).
- The apparent advantage of coupling is fully explained by per-turn generation volume, not inter-agent independence.
- Recovery follows a bimodal pattern: trajectories either re-lock within 3 exchanges, or remain captured by the perturbation for the full window.

## Experimental Conditions
1. **Coupled (C)**: Two model instances in a mutual-feeding loop, 2 generations per exchange
2. **Single (S)**: One instance self-continues, 1 generation per exchange
3. **Context-matched single (X)**: One instance self-continues twice per exchange (volume-matched control)
4. **Dose-4 (D4)**: One instance self-continues 4 times per exchange (higher-volume test)

## Reproduction
All experiments run on free-tier Google Colab with a T4 GPU.
- Each condition completes in < 25 minutes
- Total compute cost: < 3 T4 GPU-hours
- Greedy decoding + fixed openers → deterministic trajectories, fully reproducible

Steps:
1. Open Google Colab, set runtime to T4 GPU
2. Run the corresponding notebook in `/code` top-to-bottom
3. Raw outputs will match `/data/raw_outcomes.txt` exactly

## Resources
- **Full paper (PDF)**: [`paper/` directory](./paper/)
- **Permanent academic archive with DOI**: [Zenodo - 10.5281/zenodo.21157629](https://doi.org/10.5281/zenodo.21157629)
- **arXiv page**: To be updated after endorsement

## Citation