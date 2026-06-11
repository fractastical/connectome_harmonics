# Connectome Harmonics

An interactive investigation of **connectome harmonics** — graph-Laplacian eigenmodes of the brain — run on real human data, with a live simulator to explore them.

**Live demo:** https://fractastical.github.io/connectome_harmonics/

![Harmonic mode animation](connectome_harmonics_preview.gif)

---

## What this tests & what we found

Built on real data: the **HCP-YA group-average Schaefer-400 structural connectome**, a **12-subject LSD fMRI cohort** ([OpenNeuro ds003059](https://openneuro.org/datasets/ds003059)), and an independent **7-subject psilocybin cohort** for replication ([OpenNeuro ds006072](https://openneuro.org/datasets/ds006072)). Each result is reproducible with the scripts here and inspectable in the live demo's "What this demonstrates" panel.

| # | Question | Result |
|---|----------|--------|
| 1 | **Does LSD reorganize the brain's harmonic spectrum?** | **Yes (in direction).** Low-frequency harmonics lose power under LSD (LSD/placebo ≈ **0.77**), matching [Atasoy 2017](https://www.nature.com/articles/s41598-017-17546-0). The high-frequency *increase* doesn't separate at parcellated scale. |
| 2 | **Is there a geometric "latent space" in the brain?** | **Yes.** The 7 Yeo networks separate into a continuous layout in harmonic (eigenmap) coordinates — emerging from connectivity alone, with no labels given to the algorithm. |
| 3 | **Shape vs wiring — which explains activity, and does LSD change it?** | **A coarse tie; geometry ≳ wiring** (distance-only EDR basis best overall, à la [Pang 2023](https://www.nature.com/articles/s41586-023-06098-1)). Novel angle: under LSD the **geometric basis gains on the connectome** in a **within-subject paired test** — Δ ≈ +0.006, 95% CI [+0.003, +0.009], paired t(11) = 4.6, **p = 0.0008** (Wilcoxon p = 0.001, exact permutation p = 0.001), Cohen's d_z = 1.3 (large), 12/12 subjects. **Survives nuisance regression** (GSR + detrend + DVARS censoring): attenuates to Δ ≈ +0.004 but stays significant (p = 0.002, perm p = 0.003) — not merely a motion/global-signal artifact. |
| 3b | **Does that LSD shift replicate in an independent psychedelic dataset?** | **Yes, in direction; partial on significance.** A second, independent cohort — psilocybin vs active placebo (methylphenidate), [OpenNeuro ds006072](https://openneuro.org/datasets/ds006072), n = 7, fsLR-surface CIFTI → Schaefer-400 — shows the **same geometry-gains-under-drug direction**: Δ = +0.010, Cohen's d_z = 0.87 (large), **significant by exact sign-flip permutation and Wilcoxon (both p = 0.031)**, borderline by paired t (p = 0.061). Attenuates and drops below threshold under aggressive GSR (Δ = +0.006, perm p = 0.063). Different drug, different scanner, different preprocessing — so the converging direction is meaningful, even though n = 7 with an active placebo is underpowered. |

> **Scope & honesty:** results are **group-average and parcellated (400 regions)** — real data, but coarse-grained, not individual or diagnostic, and not medical advice. The geometry-vs-connectivity result is illustrative, not a settlement of that debate (cf. [Mansour 2024](https://www.biorxiv.org/content/10.1101/2024.04.16.589843v1)); the decisive test needs vertex-resolution surfaces.

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
| `compare_bases.py` | Geometry vs connectivity reconstruction (Pang 2023, parcellated) |
| `replicate_psilocybin.py` | Independent replication of the LSD shift on psilocybin (ds006072) |
| `chap_compat.py` | CHAP-compatible harmonic projection math |
| `lsd_results/` | Precomputed LSD spectra + basis comparison |
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

## Geometry vs connectivity (`compare_bases.py`)

A parcellated reproduction of [Pang et al. 2023](https://www.nature.com/articles/s41586-023-06098-1) ("Geometric constraints on human brain function"), run on the ds003059 cohort so it can also ask a question the paper did not: **does the best basis shift on LSD?**

It reconstructs the BOLD from the first *N* vectors of three Schaefer-400 bases and measures accuracy vs *N*:

- **connectome** — graph-Laplacian eigenmodes of HCP structural connectivity
- **geometric** — Laplace–Beltrami modes of the fsLR-32k cortical surface ([LaPy](https://github.com/Deep-MI/LaPy)), parcel-averaged
- **EDR** — exponential-distance-rule surrogate from parcel centroids

```bash
.venv-lsd/bin/pip install -r requirements-lsd.txt   # adds lapy
.venv-lsd/bin/python compare_bases.py               # reuses cached ds003059 BOLD
```

Outputs: `lsd_results/basis_comparison.json` + `.png` (rendered as the interactive "Geometry vs connectivity" card).

**Result (12 subjects, Schaefer-400):** all three bases are close (a coarse-parcellation tie); the distance-only **EDR** basis has the best area-under-curve in both conditions — geometry does at least as well as wiring.

**Novel angle — defensible via a within-subject paired test.** ds003059 is within-subject (every subject has both LSD and placebo), so for each subject we compute the *basis advantage* AUC(geometric) − AUC(connectome) in each state and test the LSD − placebo shift across subjects:

| Statistic | Value |
|-----------|-------|
| Mean shift Δ (LSD − placebo) | **+0.0058** |
| 95% CI | [+0.0030, +0.0085] |
| Paired *t*(11) | 4.60 |
| *p* (paired *t*) | **0.0008** |
| *p* (Wilcoxon signed-rank) | 0.0010 |
| Cohen's d_z | 1.33 (large) |
| Direction | geometry gains in **12 / 12** subjects |

So the shift is not just a pooled curiosity: under LSD, activity reliably moves toward the smoother **geometric (shape)** basis and away from specific **wiring**, consistently within every subject. The third panel of `basis_comparison.png` shows the per-subject paired lines. Numbers live in `basis_comparison.json` under `summary.paired`.

**Does it survive confound control?** Yes. The raw dataset has no fmriprep motion parameters, so we apply the strongest nuisance model computable from the parcel time series — detrend (linear+quadratic) + **global-signal regression** (the dominant motion/arousal confound for drug-state comparisons) + **DVARS frame censoring** — and rerun the identical paired test, plus an **exact within-subject sign-flip permutation** null (all 2¹² = 4096 assignments).

| | Δ (LSD − placebo) | paired *p* | permutation *p* | d_z |
|---|---|---|---|---|
| Raw | +0.0058 | 0.0008 | 0.0010 | 1.33 |
| Nuisance-regressed (98% frames kept) | +0.0044 | 0.0017 | 0.0029 | 1.19 |

The effect attenuates by ~25% under GSR (so part of the raw signal was global-signal-related) but stays significant with a large effect size — it is **not merely a motion or global-signal artifact**. Robustness numbers live in `basis_comparison.json` under `summary.robustness`.

Honest residual limits: no true (24-parameter) motion regression is possible without fmriprep derivatives; after denoising neither basis significantly *beats* the other within a single state (the claim is about the *shift*, not the absolute winner); single cohort, n = 12, parcellated.

> ⚠️ Parcellated resolution is exactly where the geometric advantage is *weakest*, and geometric modes here are hemisphere-separable. This illustrates the method and the LSD question; it does **not** settle the geometry-vs-connectivity debate (cf. [Mansour et al. 2024](https://www.biorxiv.org/content/10.1101/2024.04.16.589843v1)). The decisive test needs vertex-resolution surfaces and a carefully built connectome.

### Independent replication: psilocybin (`replicate_psilocybin.py`)

The strongest check on the LSD shift is whether it appears in a **different psychedelic, cohort, scanner, and pipeline**. We reran the identical paired test on [OpenNeuro ds006072](https://openneuro.org/datasets/ds006072) (Siegel et al. 2025) — a within-subject **psilocybin (25 mg) vs active placebo (methylphenidate 40 mg)** precision-imaging trial. These are fsLR-32k surface CIFTI dtseries (bandpassed, noGSR), parcellated to Schaefer-400 and analyzed with the same bases and statistics.

```bash
.venv-lsd/bin/python replicate_psilocybin.py   # downloads CIFTIs (cached), parcellates, paired test
```

Outputs: `lsd_results/psilocybin_replication.json` + `.png`.

| | Δ (PSIL − MTP) | paired *t* (p) | Wilcoxon *p* | permutation *p* | d_z |
|---|---|---|---|---|---|
| Raw | **+0.0103** | 2.30 (0.061) | **0.031** | **0.031** | 0.87 |
| Nuisance-regressed (94% frames kept) | +0.0061 | 1.68 (0.145) | 0.078 | 0.063 | 0.63 |

**Verdict — directional replication.** Psilocybin shifts reconstruction toward the **geometric** basis in the same direction as LSD, with a large effect (d_z = 0.87). On raw data the effect is **significant by the exact sign-flip permutation test and Wilcoxon (both p = 0.031)** — the appropriate nonparametric tests at this sample size — and borderline by the parametric *t*-test (p = 0.061). It attenuates and slips just below threshold under aggressive global-signal regression (perm p = 0.063), as the LSD effect also did. With **n = 7 against an active stimulant placebo**, the exact-permutation p-value floor is 2/128 ≈ 0.016, so the design is inherently low-powered; the converging *direction and effect size across two independent psychedelics* is the meaningful result, not a single p-value.

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
| [Siegel et al. 2025](https://openneuro.org/datasets/ds006072) | Psilocybin precision-imaging dataset (OpenNeuro ds006072) — replication cohort |
| [Pang et al. 2023](https://www.nature.com/articles/s41586-023-06098-1) | Geometric constraints on brain function (geometry vs connectome) |
| [Mansour et al. 2024](https://www.biorxiv.org/content/10.1101/2024.04.16.589843v1) | Rebuttal — connectome construction drives the comparison |
| [CHAP](https://github.com/HopkinsPsychedelic/connectome_harmonic_core) | Full vertex-level connectome harmonic pipeline |
| [NSBLab/BrainEigenmodes](https://github.com/NSBLab/BrainEigenmodes) | Pang 2023 surfaces, parcellations, eigenmode tools |

---

## Requirements

Dependency versions are **pinned** (`==`) to the exact versions used to generate the published results, for reproducibility.

| File | Packages |
|------|----------|
| `requirements.txt` | `numpy==2.4.6`, `scipy==1.17.1` — sim + data builder |
| `requirements-lsd.txt` | above + `matplotlib==3.10.9`, `nibabel==5.4.2`, `nilearn==0.13.1`, `lapy==1.6.0` — LSD + geometry analysis |

---

## License & versioning

- **Code:** [MIT](LICENSE). Reuse, modify, and redistribute freely with attribution.
- **Data:** the MIT license covers source code only. Datasets and assets downloaded at runtime (HCP, OpenNeuro ds003059, Schaefer/Yeo parcellations, NSBLab surfaces) retain their own licenses — see the note in [`LICENSE`](LICENSE) and comply with each source before redistributing derived data.
- **Releases:** tagged with [semantic versioning](https://semver.org/) (e.g. `v0.1.0`). Cite a specific tag for a stable reference to the exact code state behind a result.

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
