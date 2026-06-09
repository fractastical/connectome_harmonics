# Connectome Harmonics

An interactive educational simulator for **graph Laplacian harmonics** on brain-like networks — the mathematical framework behind [connectome-specific harmonic waves](https://www.nature.com/articles/ncomms10340) and their reorganization under [LSD](https://www.nature.com/articles/s41598-017-17546-0).

**Live demo:** https://fractastical.github.io/connectome_harmonics/

![Harmonic mode animation](connectome_harmonics_preview.gif)

---

## What is this?

The brain's structural wiring (the **connectome**) can be treated as a weighted graph. The **graph Laplacian** turns that graph into a kind of geometry. Its eigenvectors are **connectome harmonics** — spatial patterns that the network "allows" as standing waves.

| Mode | Spatial scale | Under LSD (Atasoy 2017) |
|------|---------------|-------------------------|
| Low (1–8) | Broad, bilateral | ↓ less power |
| Mid (9–25) | Regional | Mixed |
| High (26–40) | Fine-grained | ↑ more power |

This repo lets you **see** those modes oscillate on a brain-shaped graph, switch between a toy model and real HCP data, tour modes with explanations, and (optionally) reproduce a parcellated LSD vs placebo harmonic analysis from [OpenNeuro ds003059](https://openneuro.org/datasets/ds003059).

> **Disclaimer:** Educational visualization only. Not medical data about any individual, not clinical advice.

---

## Features

- **Interactive browser sim** (`index.html`) — single harmonic, superposition, Kuramoto sync
- **Mode tour** — auto-cycle modes 1–40 with plain-language explanations + LSD context
- **Data toggle** — synthetic toy graph ↔ HCP Schaefer-400 structural connectome
- **Literature sidebar** — key papers with links
- **LSD results plot** — parcellated CHAP-style spectra (demo cohort)
- **Python tools** — build harmonics from real SC, matplotlib animation, full LSD pipeline

---

## Quick start

### Browser (recommended)

Open [index.html](index.html) locally, or visit the [live demo](https://fractastical.github.io/connectome_harmonics/).

Click **start tour** to walk through harmonic modes. Toggle **Connectome data** to compare toy vs real HCP structural connectivity.

### Python animation

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# Toy synthetic graph
.venv/bin/python connectome_harmonics_sim.py --mode 7

# Real HCP Schaefer-400 structural connectome
.venv/bin/python connectome_harmonics_sim.py \
  --data connectome_harmonics_data_hcp.json --mode 7
```

---

## Project layout

| Path | Description |
|------|-------------|
| `index.html` | Main web app (GitHub Pages entry point) |
| `connectome_harmonics_simulation.html` | Original standalone sim (embedded toy data) |
| `connectome_harmonics_data.json` | Synthetic two-hemisphere toy graph |
| `connectome_harmonics_data_hcp.json` | HCP-YA group-average Schaefer-400 SC harmonics |
| `build_connectome_data.py` | Download HCP SC → compute Laplacian modes → JSON |
| `analyze_lsd_harmonics.py` | OpenNeuro ds003059 → harmonic power spectra |
| `chap_compat.py` | CHAP-compatible harmonic projection math |
| `lsd_results/` | Precomputed LSD vs placebo spectra (demo) |
| `.github/workflows/pages.yml` | GitHub Pages deploy on push to `main` |

---

## Theory (30 seconds)

Given a symmetric connectivity matrix **W**, the normalized graph Laplacian is:

```
L = I - D^{-1/2} W D^{-1/2}
```

where **D** is the degree matrix. Solve `L v_k = λ_k v_k`. Each eigenvector `v_k` is a harmonic mode; small `λ_k` = low spatial frequency (global), large `λ_k` = high frequency (local).

A standing wave on mode *k*:

```
x_i(t) = a_k · v_{k,i} · cos(√λ_k · t)
```

Real fMRI activity can be decomposed into these basis patterns — that is the core idea of [Atasoy et al. 2016](https://www.nature.com/articles/ncomms10340).

---

## Build real connectome harmonics

Uses group-average **HCP Young Adult** structural connectivity (Schaefer-400) from [GNN_SC_FC](https://github.com/PeiyuChen2023/GNN_SC_FC):

```bash
.venv/bin/python build_connectome_data.py
# → connectome_harmonics_data_hcp.json
```

Options:

```bash
.venv/bin/python build_connectome_data.py --source netneurolab
.venv/bin/python build_connectome_data.py --source mica --sc-path ./my_sc.npy
.venv/bin/python build_connectome_data.py --modes 40 --edge-percentile 95
```

---

## LSD harmonic analysis

Parcellated approximation of the [CHAP](https://github.com/HopkinsPsychedelic/connectome_harmonic_core) pipeline:

1. **Harmonics** from HCP structural connectome (`connectome_harmonics_data_hcp.json`)
2. **fMRI** from [OpenNeuro ds003059](https://openneuro.org/datasets/ds003059) (Carhart-Harris LSD study)
3. **Project** parcel BOLD onto harmonic basis → compare LSD vs placebo power spectra

```bash
python3 -m venv .venv-lsd
.venv-lsd/bin/pip install -r requirements-lsd.txt

# Quick demo (~700 MB download, 1 subject)
.venv-lsd/bin/python analyze_lsd_harmonics.py --max-subjects 1

# Larger cohort
.venv-lsd/bin/pip install -r requirements-lsd.txt
.venv-lsd/bin/python analyze_lsd_harmonics.py --max-subjects 4

# Full analysis (~10 GB, 12 subjects)
.venv-lsd/bin/python analyze_lsd_harmonics.py
```

Outputs: `lsd_results/lsd_harmonic_spectra.json` + `.png`

**Demo result (1 subject):** low-band LSD/placebo ratio ≈ 0.84, high-band ≈ 1.12 — consistent with [Atasoy 2017](https://www.nature.com/articles/s41598-017-17546-0) (↓ low-frequency, ↑ high-frequency under LSD).

For publication-faithful vertex-level harmonics, use the full [CHAP Docker pipeline](https://github.com/HopkinsPsychedelic/connectome_harmonic_core) with structural + diffusion MRI.

---

## Deploy (GitHub Pages)

**Target:** `fractastical/connectome_harmonics` → https://fractastical.github.io/connectome_harmonics/

```bash
# 1. Create empty public repo: github.com/fractastical/connectome_harmonics
# 2. Push
./push_fractastical.sh
# 3. Repo Settings → Pages → Source: GitHub Actions
```

Every push to `main` runs `.github/workflows/pages.yml` and publishes the static site.

A mirror exists at [cimcai/connectome_harmonics](https://github.com/cimcai/connectome_harmonics); the `cimc.ai` org custom domain can swallow project subpaths — **fractastical is the reliable URL**.

---

## References

| Paper | Topic |
|-------|-------|
| [Atasoy et al. 2016](https://www.nature.com/articles/ncomms10340) | Connectome-specific harmonic waves (theory) |
| [Atasoy et al. 2017](https://www.nature.com/articles/s41598-017-17546-0) | Harmonic repertoire reorganization under LSD |
| [Sanchez et al. 2020](https://www.sciencedirect.com/science/article/pii/S1053811920308508) | Robustness of harmonics to connectivity changes |
| [Vohryzek et al. 2024](https://www.nature.com/articles/s42003-024-06669-6) | Integrative / segregative / degenerate harmonics |
| [Carhart-Harris et al. 2020](https://openneuro.org/datasets/ds003059) | LSD fMRI dataset (OpenNeuro ds003059) |
| [CHAP](https://github.com/HopkinsPsychedelic/connectome_harmonic_core) | Full vertex-level connectome harmonic pipeline |

---

## Requirements

| File | Packages |
|------|----------|
| `requirements.txt` | numpy, scipy, matplotlib — sim + data builder |
| `requirements-lsd.txt` | above + nibabel, nilearn — LSD analysis |

---

## Citation

If you use this educational repo, please cite the underlying methods:

```bibtex
@article{atasoy2016connectome,
  title={Human brain networks function in connectome-specific harmonic waves},
  author={Atasoy, Selen and Donnelly, Isaac and Pearson, Joel},
  journal={Nature Communications},
  volume={7},
  pages={10340},
  year={2016}
}

@article{atasoy2017lsd,
  title={Connectome-harmonic decomposition of human brain activity reveals dynamical repertoire re-organization under {LSD}},
  author={Atasoy, Selen and Roseman, Leor and Kaelen, Mendel and others},
  journal={Scientific Reports},
  volume={7},
  pages={17661},
  year={2017}
}
```
