"""
Harmonic consonance / spectral-symmetry under psychedelics (STV probe).

Motivation
----------
The Symmetry Theory of Valence (STV; Johnson 2016, *Principia Qualia*) proposes
that the pleasantness (valence) of an experience tracks the *symmetry* of the
mathematical object isomorphic to it. Gomez-Emilsson and the Qualia Research
Institute operationalised this for the brain using Atasoy's connectome harmonics:
the candidate "object" is the harmonic power spectrum, and its *consonance* (the
degree to which the active harmonics stand in simple, mutually reinforcing
frequency relationships, as in musical consonance) is the proposed valence proxy.

We cannot test the valence correlation directly here -- neither ds003059 (LSD)
nor ds006072 (psilocybin) ships per-subject positive-affect ratings. So we test
the *mechanism half* of the hypothesis that does not need affect labels:

    Does a serotonergic psychedelic move the harmonic spectrum toward greater
    consonance / symmetry, or away from it?

This matters because two effects in our own data point in opposite STV directions:
activity shifts toward the smooth geometric basis (more "aligned with harmonic
geometry" -> STV-positive), yet the spectrum broadens / desynchronises
(Atasoy 2017; Siegel 2024 -> naively STV-negative). A consonance metric
disambiguates them.

What we compute (per subject x condition)
-----------------------------------------
Project z-scored parcel BOLD onto a harmonic basis (connectome harmonics =
graph-Laplacian eigenmodes; primary substrate, orthonormal as in CHAP). Each
mode k has eigenvalue lambda_k -> spatial frequency omega_k = sqrt(lambda_k) and
relative power p_k (Sum p_k = 1). From {omega_k, p_k} we derive:

  * sethares_dissonance : Sethares (1993) sensory-dissonance of the spectrum
                          treated as a chord (omega mapped proportionally to an
                          audio range so ratios are preserved). LOWER = more
                          consonant.  consonance = 1/(1 + dissonance).
  * spectral_entropy    : normalised Shannon entropy of p (broader repertoire).
  * spectral_centroid   : Sum p_k omega_k (energy toward high spatial freq).
  * low_high_ratio      : power in lowest third / power in highest third.

Then a within-subject paired test of (drug - placebo) for each metric, with an
exact sign-flip permutation null and a GSR robustness rerun.

STV reading: psychedelic -> LOWER dissonance (higher consonance) would support an
STV/"neural annealing" account; psychedelic -> HIGHER dissonance/entropy is the
opposite. We report the direction honestly either way.

Caveats: n = 12 (LSD) / 7 (psilocybin), parcellated (400 regions); the omega->Hz
mapping is a modelling choice (we report robustness to it); consonance is a
*proxy* for STV-symmetry, not a measurement of valence.

Run:
    .venv-lsd/bin/python harmonic_consonance.py            # both datasets
    .venv-lsd/bin/python harmonic_consonance.py --dataset psilocybin
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from compare_bases import (
    _hemi_geometric,
    _zscore_run,
    assemble_maps,
    paired_stats,
    regress_nuisance,
    signflip_perm,
    subject_runs,
)

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "lsd_results"
PSIL_PARC = ROOT / ".cache" / "ds006072" / "parc"

# Reference: which on-drug session is psilocybin vs methylphenidate per subject.
PSIL_CONDS = ("psil", "mtp")           # (drug, placebo)
LSD_CONDS = ("lsd", "placebo")         # (drug, placebo)

METRICS = ["sethares_dissonance", "spectral_entropy", "spectral_centroid",
           "low_high_ratio"]
METRIC_LABELS = {
    "sethares_dissonance": "Sethares dissonance\n(lower = more consonant)",
    "spectral_entropy": "spectral entropy\n(higher = broader)",
    "spectral_centroid": "spectral centroid\n(higher spatial freq)",
    "low_high_ratio": "low/high power ratio\n(higher = more low-freq)",
}


# ---------------------------------------------------------------------------
# Harmonic bases (with eigenvalues -> spatial frequencies)
# ---------------------------------------------------------------------------
def connectome_modes(n_modes: int) -> tuple[np.ndarray, np.ndarray]:
    """Graph-Laplacian eigenmodes of HCP SC: (omega [n_modes], V [400, n_modes])."""
    from build_connectome_data import load_sc_matrix, normalized_laplacian_modes

    w = load_sc_matrix("gnn_scfc", None)
    evals, evecs = normalized_laplacian_modes(w, n_modes)
    return np.sqrt(np.maximum(evals, 0.0)), evecs


def geometric_modes(n_modes: int) -> tuple[np.ndarray, np.ndarray]:
    """LBO cortical-geometry eigenmodes, parcel-averaged, ordered by frequency."""
    lh_parc, lh_evals, lh_ids = _hemi_geometric(
        "fsLR_32k_midthickness-lh.vtk", "fsLR_32k_Schaefer400-lh.txt", n_modes)
    rh_parc, rh_evals, rh_ids = _hemi_geometric(
        "fsLR_32k_midthickness-rh.vtk", "fsLR_32k_Schaefer400-rh.txt", n_modes)
    entries = []
    for k in range(n_modes):
        col = np.zeros(400); col[lh_ids - 1] = lh_parc[:, k]
        entries.append((lh_evals[k], col))
        col = np.zeros(400); col[rh_ids - 1] = rh_parc[:, k]
        entries.append((rh_evals[k], col))
    entries.sort(key=lambda e: e[0])
    evals = np.array([e for e, _ in entries[:n_modes]])
    V = np.column_stack([c for _, c in entries[:n_modes]])
    V = V / np.maximum(np.linalg.norm(V, axis=0, keepdims=True), 1e-12)
    return np.sqrt(np.maximum(evals, 0.0)), V


# ---------------------------------------------------------------------------
# Spectrum + consonance metrics
# ---------------------------------------------------------------------------
def power_spectrum(V: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Relative harmonic power p_k of (z-scored) BOLD Y (400, T) on basis V."""
    Yc = Y - Y.mean(axis=0, keepdims=True)        # spatially center each frame
    C = V.T @ Yc                                   # (n_modes, T) mode coefficients
    p = (C ** 2).mean(axis=1)                      # time-averaged power per mode
    s = p.sum()
    return p / s if s > 0 else p


def sethares_dissonance(freqs: np.ndarray, amps: np.ndarray) -> float:
    """Total sensory dissonance of a 'chord' (Sethares 1993, vectorised).

    freqs in Hz, amps are linear amplitudes. Lower = more consonant.
    """
    Dstar, S1, S2 = 0.24, 0.0207, 18.96
    A1, A2, C1, C2 = 3.51, 5.75, 5.0, -5.0
    f = np.asarray(freqs, float)
    a = np.asarray(amps, float)
    Fmin = np.minimum(f[:, None], f[None, :])
    Fdif = np.abs(f[:, None] - f[None, :])
    amin = np.minimum(a[:, None], a[None, :])
    S = Dstar / (S1 * Fmin + S2)
    d = amin * (C1 * np.exp(-A1 * S * Fdif) + C2 * np.exp(-A2 * S * Fdif))
    return float(np.triu(d, k=1).sum())


def spectrum_metrics(omega: np.ndarray, p: np.ndarray, f_low: float = 220.0) -> dict:
    """All scalar metrics from spatial frequencies omega and relative power p."""
    K = p.size
    eps = 1e-12
    # Map omega proportionally to an audio range (ratios preserved); lowest -> f_low.
    wpos = np.maximum(omega, omega[omega > 0].min() if np.any(omega > 0) else 1.0)
    f_hz = f_low * wpos / wpos.min()
    diss = sethares_dissonance(f_hz, np.sqrt(p))
    entropy = float(-(p * np.log(p + eps)).sum() / np.log(K))
    centroid = float((p * omega).sum())
    third = max(K // 3, 1)
    low = float(p[:third].sum())
    high = float(p[-third:].sum())
    return {
        "sethares_dissonance": diss,
        "consonance": float(1.0 / (1.0 + diss)),
        "spectral_entropy": entropy,
        "spectral_centroid": centroid,
        "low_high_ratio": float(low / max(high, eps)),
    }


# ---------------------------------------------------------------------------
# Per-subject map loaders (reuse cached data; no network)
# ---------------------------------------------------------------------------
def psil_maps(denoise: bool) -> dict[str, dict[str, np.ndarray]]:
    """{P<n>: {psil, mtp}} z-scored parcel maps from the cached .npy parcels."""
    maps: dict[str, dict[str, np.ndarray]] = {}
    for sub in range(1, 8):
        entry = {}
        ok = True
        for cond in PSIL_CONDS:
            f = PSIL_PARC / f"sub-{sub}_{cond}.npy"
            if not f.exists():
                ok = False
                break
            ts = np.load(f)                          # (400, T)
            if denoise:
                ts = regress_nuisance(ts)
            entry[cond] = _zscore_run(ts)
        if ok:
            maps[f"P{sub}"] = entry
    return maps


def lsd_maps(denoise: bool, max_subjects: int | None) -> dict[str, dict[str, np.ndarray]]:
    """{sub: {lsd, placebo}} z-scored parcel maps from cached ds003059 volumes."""
    from analyze_lsd_harmonics import (
        ALL_SUBJECTS, EXCLUDE_SUBJECTS, fetch_schaefer_masker,
    )

    subjects = [s for s in ALL_SUBJECTS if s not in EXCLUDE_SUBJECTS]
    if max_subjects:
        subjects = subjects[:max_subjects]
    masker = fetch_schaefer_masker()
    maps: dict[str, dict[str, np.ndarray]] = {}
    for sub in subjects:
        entry = {}
        ok = True
        for cond in LSD_CONDS:
            try:
                runs = subject_runs(sub, cond, masker)
            except Exception as exc:  # noqa: BLE001
                print(f"  ! {sub} {cond}: {exc}")
                runs = []
            m, _ = assemble_maps(runs, denoise=denoise)
            if m is None:
                ok = False
                break
            entry[cond] = m
        if ok:
            maps[sub] = entry
    return maps


# ---------------------------------------------------------------------------
# Analysis driver
# ---------------------------------------------------------------------------
def analyse(maps: dict, conds: tuple[str, str], omega: np.ndarray, V: np.ndarray) -> dict:
    """Per-subject metrics + paired drug-vs-placebo tests for one basis."""
    drug, placebo = conds
    subjects = sorted(maps)
    per = {drug: {m: [] for m in METRICS}, placebo: {m: [] for m in METRICS}}
    per_extra = {drug: {"consonance": []}, placebo: {"consonance": []}}
    for s in subjects:
        for cond in conds:
            p = power_spectrum(V, maps[s][cond])
            met = spectrum_metrics(omega, p)
            for m in METRICS:
                per[cond][m].append(met[m])
            per_extra[cond]["consonance"].append(met["consonance"])

    tests = {}
    for m in METRICS + ["consonance"]:
        a = (per[drug][m] if m in METRICS else per_extra[drug]["consonance"])
        b = (per[placebo][m] if m in METRICS else per_extra[placebo]["consonance"])
        st = paired_stats(a, b)
        st["perm"] = signflip_perm([x - y for x, y in zip(a, b)])
        tests[m] = st

    return {
        "subjects": subjects,
        "per_subject": {
            cond: {**per[cond], "consonance": per_extra[cond]["consonance"]}
            for cond in conds
        },
        "paired_drug_minus_placebo": tests,
    }


def run_dataset(name: str, maps_raw: dict, maps_dn: dict, conds: tuple[str, str],
                bases: dict) -> dict:
    out = {"conditions": {"drug": conds[0], "placebo": conds[1]},
           "n_subjects": len(maps_raw), "bases": {}}
    for bname, (omega, V) in bases.items():
        raw = analyse(maps_raw, conds, omega, V)
        dn = analyse(maps_dn, conds, omega, V)
        out["bases"][bname] = {"raw": raw, "denoised": dn}
    return out


def stv_reading(diss_test: dict) -> str:
    """One-line STV interpretation from the dissonance drug-minus-placebo test."""
    d = diss_test["mean_diff"]      # drug - placebo dissonance
    p = diss_test["p_ttest"]
    direction = ("LOWER dissonance under drug (more consonant -> STV-positive)"
                 if d < 0 else
                 "HIGHER dissonance under drug (less consonant -> STV-negative)")
    sig = "significant" if p < 0.05 else ("trend" if p < 0.1 else "n.s.")
    return f"{direction} [{sig}, p={p:.3f}]"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", choices=["both", "lsd", "psilocybin"], default="both")
    ap.add_argument("--modes", type=int, default=200)
    ap.add_argument("--max-subjects", type=int, default=None)
    ap.add_argument("--output-dir", type=Path, default=OUT)
    args = ap.parse_args()

    print(f"Building harmonic bases (M={args.modes}) …")
    bases = {
        "connectome": connectome_modes(args.modes),
        "geometric": geometric_modes(args.modes),
    }

    payload = {
        "meta": {
            "question": ("Does a serotonergic psychedelic move the harmonic power "
                         "spectrum toward consonance/symmetry (STV) or away from it?"),
            "primary_basis": "connectome harmonics (graph-Laplacian eigenmodes, HCP SC)",
            "n_modes": args.modes,
            "freq_mapping": "omega_k = sqrt(lambda_k), mapped proportionally to audio Hz (lowest -> 220 Hz)",
            "metrics": METRICS + ["consonance"],
            "stv_reference": "Johnson 2016 (Principia Qualia); Gomez-Emilsson / QRI harmonic consonance",
            "caveats": ("n=12 (LSD)/7 (psilocybin), parcellated (Schaefer-400); "
                        "consonance is an STV proxy, not a valence measurement; "
                        "no per-subject affect ratings available in either release."),
        },
        "datasets": {},
    }

    if args.dataset in ("both", "psilocybin"):
        print("PSILOCYBIN (ds006072): loading cached parcels …")
        ds = run_dataset("psilocybin", psil_maps(False), psil_maps(True),
                         PSIL_CONDS, bases)
        ds["meta"] = {"dataset": "OpenNeuro ds006072 (psilocybin vs methylphenidate)",
                      "reference": "Siegel et al. 2025"}
        payload["datasets"]["psilocybin"] = ds

    if args.dataset in ("both", "lsd"):
        print("LSD (ds003059): extracting parcels from cached volumes …")
        ds = run_dataset("lsd", lsd_maps(False, args.max_subjects),
                         lsd_maps(True, args.max_subjects), LSD_CONDS, bases)
        ds["meta"] = {"dataset": "OpenNeuro ds003059 (LSD vs placebo)",
                      "reference": "Carhart-Harris et al. 2016 / 2020"}
        payload["datasets"]["lsd"] = ds

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_json = args.output_dir / "harmonic_consonance.json"
    out_json.write_text(json.dumps(payload, indent=2))
    plot_results(payload, args.output_dir / "harmonic_consonance.png")

    print(f"\nWrote {out_json}\n")
    print("=" * 78)
    print("STV READING (connectome harmonics, raw):")
    for dname, ds in payload["datasets"].items():
        tests = ds["bases"]["connectome"]["raw"]["paired_drug_minus_placebo"]
        diss = tests["sethares_dissonance"]
        cons = tests["consonance"]
        ent = tests["spectral_entropy"]
        print(f"\n  {dname.upper()} (n={ds['n_subjects']}):")
        print(f"    dissonance Δ={diss['mean_diff']:+.4f} p={diss['p_ttest']:.3f} "
              f"perm={diss['perm']['p']:.3f} d_z={diss['cohen_dz']:.2f}")
        print(f"    consonance Δ={cons['mean_diff']:+.4f} p={cons['p_ttest']:.3f} "
              f"perm={cons['perm']['p']:.3f}")
        print(f"    entropy    Δ={ent['mean_diff']:+.4f} p={ent['p_ttest']:.3f}")
        print(f"    -> {stv_reading(diss)}")
    print("=" * 78)


def plot_results(payload: dict, out_png: Path) -> None:
    dsets = list(payload["datasets"])
    metrics = ["sethares_dissonance", "spectral_entropy", "spectral_centroid"]
    nrow, ncol = len(metrics), max(len(dsets), 1)
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.0 * ncol, 3.4 * nrow),
                             squeeze=False)
    fig.patch.set_facecolor("#0b0e14")
    for ci, dname in enumerate(dsets):
        ds = payload["datasets"][dname]
        drug, placebo = ds["conditions"]["drug"], ds["conditions"]["placebo"]
        res = ds["bases"]["connectome"]["raw"]
        dn = ds["bases"]["connectome"]["denoised"]
        for ri, metric in enumerate(metrics):
            ax = axes[ri][ci]
            ax.set_facecolor("#0b0e14")
            for sp in ax.spines.values():
                sp.set_color("#333")
            ax.tick_params(colors="#9fb0c8")
            gp = res["per_subject"][placebo][metric]
            gd = res["per_subject"][drug][metric]
            # All three plotted metrics are "lower = more symmetric/consonant"
            # (less dissonance, narrower repertoire, lower spatial frequency),
            # so a downward move under the drug is "toward symmetry" (green).
            for a, b in zip(gp, gd):
                better = b < a
                col = "#7CE0B0" if better else "#ff9a9a"
                ax.plot([0, 1], [a, b], color=col, lw=1, alpha=0.5,
                        marker="o", ms=3, mfc=col, mec="none")
            ax.plot([0, 1], [np.mean(gp), np.mean(gd)], color="white", lw=2.5,
                    marker="o", ms=6, zorder=5, label="mean (raw)")
            gpd = dn["per_subject"][placebo][metric]
            gdd = dn["per_subject"][drug][metric]
            ax.plot([0, 1], [np.mean(gpd), np.mean(gdd)], color="#8ec5ff", lw=2.0,
                    ls=(0, (4, 2)), marker="s", ms=5, zorder=6, label="mean (GSR)")
            ax.set_xlim(-0.35, 1.35)
            ax.set_xticks([0, 1])
            ax.set_xticklabels([placebo.upper(), drug.upper()])
            st = res["paired_drug_minus_placebo"][metric]
            if ri == 0:
                ax.set_title(f"{dname.upper()}  (n={ds['n_subjects']})\n"
                             f"{METRIC_LABELS[metric].splitlines()[0]}  "
                             f"Δ={st['mean_diff']:+.3f}, p={st['p_ttest']:.3f}",
                             color="white", fontsize=9.5)
            else:
                ax.set_title(f"{METRIC_LABELS[metric].splitlines()[0]}  "
                             f"Δ={st['mean_diff']:+.3f}, p={st['p_ttest']:.3f}, "
                             f"perm={st['perm']['p']:.3f}",
                             color="white", fontsize=9.5)
            if ci == 0:
                ax.set_ylabel(METRIC_LABELS[metric], color="#cfe0f5", fontsize=9)
            if ri == 0 and ci == 0:
                ax.legend(facecolor="#111722", edgecolor="#333",
                          labelcolor="white", fontsize=8, loc="best")
    fig.suptitle("Harmonic spectral symmetry under psychedelics  ·  "
                 "connectome harmonics  ·  green = toward symmetry/consonance",
                 color="#cfe0f5", fontsize=11.5, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)


if __name__ == "__main__":
    main()
