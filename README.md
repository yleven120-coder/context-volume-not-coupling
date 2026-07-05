# Volume or Coupling? A Scale-Dependent Dissociation in Constraint Recovery of Language-Model Loops

**Author:** Simin Yuan  
**Contact:** yleven120@gmail.com  
**Zenodo (v2):** [10.5281/zenodo.21158324](https://doi.org/10.5281/zenodo.21158324)  
**arXiv:** *pending (v1 under review; v2 replacement ready)*

---

## Project Overview

This repository contains the full replication package for the paper **"Volume or Coupling? A Scale-Dependent Dissociation in Constraint Recovery of Language-Model Loops"** (v2, July 2026).

The study investigates whether coupling a language model to a second agent improves recovery from a conflicting instruction, or whether the effect is merely due to increased generation volume. We use a minimal perturbation–recovery protocol (uppercase constraint, polite lowercase perturbation, 8‑exchange window) across **four conditions** (Coupled, Single, Context‑matched single, Dose‑4) and **three model scales** (1.5B, 3B, 7B) within the Qwen2.5‑Instruct family.

**Key findings:**
- **At 1.5B (v1 base result):** Coupling advantage is fully explained by volume (C: 43%, S: 17%, X: 37%; C vs. X p=0.774).  
- **At 3B:** All conditions at ceiling (90–100%) – perturbation fails to capture.  
- **At 7B (v2 reversal):** Volume completely fails (S=0%, X=0%, D4=3.3%), but coupling uniquely recovers (C=26.7%, C vs. X p=0.0078). Recovery is slow (delays 2–7) and driven by **asymmetric capture depth**: the directly perturbed agent is deeply captured, while its shielded partner (never directly receiving the instruction) is only shallowly captured by mimicry and re‑anchors, supplying constraint‑consistent evidence.

The volume effect is therefore **scale‑bounded**: it only operates at small scales; at larger scales, structural shielding (coupling) provides volume‑irreducible recovery.

---

## Repository Structure
**How to distinguish v1 vs. v2 files:**  
All v1 files are named without scale suffixes (e.g., `run_1.5B.py`, `raw_1.5B.txt`, `fig1_setup.png`).  
All v2 additions are clearly marked with `_3B`, `_7B`, `_scale`, or `_traj` in filenames, or placed in subfolders named `v2/` where applicable. Check each folder for detailed file listings.

**Why keep v1 files (paper/ and all v1 data/code/figures)?**  
To maintain a complete, transparent research trajectory – reviewers and readers can compare the original deflationary result (v1) with the extended scale‑dependent findings (v2). This aligns with Zenodo versioning and arXiv replacement records.

---

## Getting Started

### Hardware Requirements
- **1.5B tier:** Runs on free Google Colab T4 (under 3 GPU‑hours total).
- **3B & 7B tiers:** Require L4 GPU (approx. 5 GPU‑hours combined).

### Run the Experiments
All scripts are in `code/`. The v1 script (`run_1.5B.ipynb`) reproduces the base experiment. The v2 scale‑extension scripts are named `run_3B.ipynb` and `run_7B.ipynb`. Greedy decoding ensures deterministic outputs, so re‑running will reproduce exactly the reported data.

### Reproduce Figures
The `figures/` folder contains all `.png` outputs. You can regenerate the v2 main figures by running `v2_figures.py` (provided in the root). The v1 figures are also included for reference.

---

## Citation

If you use this work, please cite the Zenodo record (concept DOI, always points to latest version):

```bibtex
@misc{yuan2026volume,
  author = {Simin Yuan},
  title = {Volume or Coupling? A Scale-Dependent Dissociation in Constraint Recovery of Language-Model Loops},
  year = {2026},
  howpublished = {arXiv preprint (v2)},
  note = {Zenodo: 10.5281/zenodo.21158324}
}
## License

This work is licensed under a [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/).

## Acknowledgments

Experiment code and manuscript drafting were assisted by an AI system (Claude, Anthropic). The author thanks Kai‑Wei Chang for arXiv endorsement and feedback on figures.
