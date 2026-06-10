"""
Geometry vs connectivity: eigenmode reconstruction showdown (parcellated).

A Schaefer-400 reproduction of the central comparison in Pang et al. 2023
(Nature, "Geometric constraints on human brain function"), run on our own
LSD data so we can also ask a question the original paper did not:

    Does the *best basis* shift between placebo and LSD?

We reconstruct ds003059 resting-state BOLD (already downloaded) from the first
N vectors of three Schaefer-400 spatial bases and measure reconstruction
accuracy vs N:

  - connectome : graph-Laplacian eigenmodes of HCP structural connectivity
  - geometric  : Laplace-Beltrami eigenmodes of the fsLR-32k cortical surface
                 (computed with LaPy), parcel-averaged to Schaefer-400
  - edr        : exponential-distance-rule surrogate from parcel centroids

Honest caveats (see README / index.html):
  * Parcellated (400 regions): this is the resolution at which Pang's advantage
    is *weakest*, and parcel-averaging blurs the fine geometric modes.
  * Geometric modes are hemisphere-separable by construction; connectome / EDR
    are whole-brain (can carry interhemispheric structure).
  * ds003059 is volumetric, projected to parcels via nilearn (not surface CIFTI).
  This illustrates the method and the LSD question; it does not adjudicate the
  geometry-vs-connectivity debate (cf. Mansour et al. 2024).

Run:
    python compare_bases.py                 # full cohort (uses cached BOLD)
    python compare_bases.py --max-subjects 2
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from analyze_lsd_harmonics import (
    ALL_SUBJECTS,
    EXCLUDE_SUBJECTS,
    collect_runs,
    download_openneuro,
    extract_parcel_timeseries,
    fetch_schaefer_masker,
)
from build_connectome_data import (
    load_schaefer_centroids,
    load_sc_matrix,
    normalized_laplacian_modes,
)

ROOT = Path(__file__).resolve().parent
PANG = ROOT / ".cache" / "pang"
OUT = ROOT / "lsd_results"

PANG_BASE = (
    "https://raw.githubusercontent.com/NSBLab/BrainEigenmodes/main/data"
)
PANG_FILES = {
    "fsLR_32k_midthickness-lh.vtk": "template_surfaces_volumes/fsLR_32k_midthickness-lh.vtk",
    "fsLR_32k_midthickness-rh.vtk": "template_surfaces_volumes/fsLR_32k_midthickness-rh.vtk",
    "fsLR_32k_Schaefer400-lh.txt": "parcellations/fsLR_32k_Schaefer400-lh.txt",
    "fsLR_32k_Schaefer400-rh.txt": "parcellations/fsLR_32k_Schaefer400-rh.txt",
}


def fetch_pang(name: str) -> Path:
    import urllib.request

    PANG.mkdir(parents=True, exist_ok=True)
    local = PANG / name
    if not local.exists():
        url = f"{PANG_BASE}/{PANG_FILES[name]}"
        print(f"Downloading {name} …")
        urllib.request.urlretrieve(url, local)
    return local


# ----------------------------------------------------------------------------
# Bases (each returns a 400 x M matrix whose first N columns span a nested
# low-frequency subspace, plus the eigenvalue ordering used).
# ----------------------------------------------------------------------------
def connectome_basis(n_modes: int) -> np.ndarray:
    w = load_sc_matrix("gnn_scfc", None)
    _, eigvecs = normalized_laplacian_modes(w, n_modes)
    return eigvecs  # (400, n_modes)


def edr_basis(n_modes: int, length_mm: float) -> np.ndarray:
    coords, _, _ = load_schaefer_centroids()
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt((diff ** 2).sum(-1))
    w = np.exp(-dist / length_mm)
    np.fill_diagonal(w, 0.0)
    _, eigvecs = normalized_laplacian_modes(w, n_modes)
    return eigvecs


def _hemi_geometric(name: str, label_name: str, n_modes: int):
    """LBO eigenmodes of one masked hemisphere, parcel-averaged.

    Returns (parc_modes [n_parcels, n_modes], eigvals [n_modes], parcel_ids).
    """
    from lapy import Solver, TriaMesh

    mesh = TriaMesh.read_vtk(str(fetch_pang(name)))
    labels = np.loadtxt(fetch_pang(label_name)).astype(int)
    keep = labels > 0

    # Restrict mesh to cortex (drop medial wall) and remap triangle indices.
    idx_map = -np.ones(labels.shape[0], dtype=int)
    idx_map[keep] = np.arange(int(keep.sum()))
    tris = mesh.t
    tmask = keep[tris].all(axis=1)
    cortex = TriaMesh(mesh.v[keep], idx_map[tris[tmask]])
    labels_keep = labels[keep]

    fem = Solver(cortex)
    evals, evecs = fem.eigs(k=n_modes + 1)
    evals = np.asarray(evals)[1:]          # drop constant mode
    evecs = np.asarray(evecs)[:, 1:]       # (n_cortex_verts, n_modes)

    parcel_ids = np.unique(labels_keep)
    parc = np.zeros((parcel_ids.size, n_modes))
    for i, pid in enumerate(parcel_ids):
        parc[i] = evecs[labels_keep == pid].mean(axis=0)
    return parc, evals, parcel_ids


def geometric_basis(n_modes: int) -> np.ndarray:
    lh_parc, lh_evals, lh_ids = _hemi_geometric(
        "fsLR_32k_midthickness-lh.vtk", "fsLR_32k_Schaefer400-lh.txt", n_modes
    )
    rh_parc, rh_evals, rh_ids = _hemi_geometric(
        "fsLR_32k_midthickness-rh.vtk", "fsLR_32k_Schaefer400-rh.txt", n_modes
    )
    # Assemble hemisphere-separable whole-brain modes (400-vectors), then order
    # all of them by spatial frequency (eigenvalue) and keep the lowest n_modes.
    entries = []
    for k in range(n_modes):
        col = np.zeros(400)
        col[lh_ids - 1] = lh_parc[:, k]   # labels are 1-based global parcel ids
        entries.append((lh_evals[k], col))
        col = np.zeros(400)
        col[rh_ids - 1] = rh_parc[:, k]
        entries.append((rh_evals[k], col))
    entries.sort(key=lambda e: e[0])
    return np.column_stack([c for _, c in entries[:n_modes]])


# ----------------------------------------------------------------------------
# Reconstruction accuracy
# ----------------------------------------------------------------------------
def reconstruction_curve(basis: np.ndarray, Y: np.ndarray, grid: list[int]) -> list[float]:
    """Mean per-frame correlation between Y and its projection onto first-N modes."""
    Q, _ = np.linalg.qr(basis)               # nested orthonormal spans
    Yc = Y - Y.mean(axis=0, keepdims=True)   # center each frame spatially
    ynorm = np.linalg.norm(Yc, axis=0)
    valid = ynorm > 1e-9
    Yc, ynorm = Yc[:, valid], ynorm[valid]
    C = Q.T @ Yc
    out = []
    for n in grid:
        R = Q[:, :n] @ C[:n]
        rnorm = np.linalg.norm(R, axis=0)
        corr = (Yc * R).sum(axis=0) / (ynorm * np.maximum(rnorm, 1e-12))
        out.append(float(np.mean(corr)))
    return out


def regress_nuisance(ts: np.ndarray) -> np.ndarray:
    """Detrend, global-signal-regress, and DVARS-censor one run.

    ts: (n_parcels, T) raw parcel BOLD -> cleaned (n_parcels, T_kept).

    This raw OpenNeuro dataset ships without fmriprep motion parameters, so we
    apply the strongest confound model computable from the parcel time series
    alone — the kind of nuisance regression a reviewer would demand before
    believing a drug-state effect is neural rather than motion/physiology:

      * constant + linear + quadratic drift     (scanner drift)
      * global signal and its temporal deriv     (whole-brain motion / arousal —
                                                  the dominant psychedelic-state
                                                  confound; this is GSR)
      * DVARS-based frame censoring (Tukey fence) as a motion-spike proxy
    """
    n, T = ts.shape
    X = ts.T.astype(float)                       # (T, n)
    tn = np.linspace(-1.0, 1.0, T)
    gs = X.mean(axis=1)                          # global signal
    gsd = np.gradient(gs)
    R = np.column_stack([np.ones(T), tn, tn ** 2, gs, gsd])
    beta, *_ = np.linalg.lstsq(R, X, rcond=None)
    Xc = X - R @ beta                            # residuals (T, n)
    dv = np.zeros(T)
    dv[1:] = np.sqrt((np.diff(Xc, axis=0) ** 2).mean(axis=1))   # DVARS
    q1, q3 = np.percentile(dv[1:], [25, 75])
    keep = dv <= (q3 + 1.5 * (q3 - q1))          # Tukey upper fence
    keep[0] = True
    if keep.sum() < max(10, 0.5 * T):            # safety: never drop > half
        keep = np.ones(T, dtype=bool)
    return Xc[keep].T                            # (n, T_kept)


def _zscore_run(ts: np.ndarray) -> np.ndarray:
    mu = ts.mean(axis=1, keepdims=True)
    sd = ts.std(axis=1, keepdims=True)
    sd[sd < 1e-9] = 1.0
    return (ts - mu) / sd


def subject_runs(subject: str, condition: str, masker) -> list[np.ndarray]:
    """Raw (n_parcels, T) parcel BOLD for each run of one subject/condition."""
    runs = []
    for spec in collect_runs([subject]):
        if spec["condition"] != condition:
            continue
        path = download_openneuro(spec["key"])
        runs.append(extract_parcel_timeseries(path, masker))
    return runs


def assemble_maps(runs: list[np.ndarray], denoise: bool) -> tuple[np.ndarray | None, float]:
    """Optionally clean, z-score per run, concatenate. Returns (map, kept_frac)."""
    cols, kept, total = [], 0, 0
    for ts in runs:
        total += ts.shape[1]
        if denoise:
            ts = regress_nuisance(ts)
        kept += ts.shape[1]
        cols.append(_zscore_run(ts))
    if not cols:
        return None, 0.0
    return np.concatenate(cols, axis=1), (kept / total if total else 0.0)


def modes_to_threshold(grid: list[int], curve: list[float], thr: float) -> float:
    for i, c in enumerate(curve):
        if c >= thr:
            if i == 0:
                return float(grid[0])
            x0, x1, y0, y1 = grid[i - 1], grid[i], curve[i - 1], curve[i]
            return float(x0 + (thr - y0) * (x1 - x0) / (y1 - y0 + 1e-12))
    return float("nan")


def build_grid(n_max: int) -> list[int]:
    raw = [1, 2, 3, 4, 5, 7, 10, 14, 20, 28, 40, 56, 80, 112, 160, 200, 280, 400]
    return sorted({min(n, n_max) for n in raw if n <= n_max})


def auc(grid: list[int], curve: list[float]) -> float:
    """Normalized area under the reconstruction-accuracy curve (mean accuracy)."""
    x, y = np.asarray(grid, float), np.asarray(curve, float)
    area = np.sum((x[1:] - x[:-1]) * (y[1:] + y[:-1]) / 2.0)
    return float(area / (x[-1] - x[0]))


def paired_stats(a: list[float], b: list[float]) -> dict:
    """Paired comparison of a vs b (e.g. per-subject gap under LSD vs placebo).

    Returns mean difference (a-b), 95% CI, paired t-test, Wilcoxon signed-rank,
    and Cohen's d_z. All across the subject dimension.
    """
    from scipy import stats

    a = np.asarray(a, float)
    b = np.asarray(b, float)
    d = a - b
    n = int(d.size)
    out = {
        "n": n,
        "mean_a": float(a.mean()) if n else float("nan"),
        "mean_b": float(b.mean()) if n else float("nan"),
        "mean_diff": float(d.mean()) if n else float("nan"),
    }
    if n < 2:
        out.update(
            sd_diff=float("nan"), ci95=[float("nan"), float("nan")],
            t=float("nan"), p_ttest=float("nan"),
            w=float("nan"), p_wilcoxon=float("nan"), cohen_dz=float("nan"),
        )
        return out
    sd = float(d.std(ddof=1))
    se = sd / np.sqrt(n)
    tcrit = float(stats.t.ppf(0.975, n - 1))
    t_res = stats.ttest_rel(a, b)
    try:
        w_res = stats.wilcoxon(a, b)
        w_stat, w_p = float(w_res.statistic), float(w_res.pvalue)
    except ValueError:  # all-zero differences
        w_stat, w_p = float("nan"), float("nan")
    out.update(
        sd_diff=sd,
        ci95=[out["mean_diff"] - tcrit * se, out["mean_diff"] + tcrit * se],
        t=float(t_res.statistic),
        p_ttest=float(t_res.pvalue),
        w=w_stat,
        p_wilcoxon=w_p,
        cohen_dz=(out["mean_diff"] / sd) if sd > 0 else float("nan"),
    )
    return out


def signflip_perm(delta: list[float]) -> dict:
    """Exact sign-flip permutation test for paired data (null = label swap).

    Under H0 the LSD/placebo label is exchangeable within each subject, i.e. the
    per-subject difference is equally likely to be +d or -d. For n<=20 we
    enumerate all 2^n sign assignments exactly; otherwise we sample. The p-value
    is the two-sided fraction of permuted means at least as extreme as observed.
    """
    import itertools

    d = np.asarray(delta, float)
    n = d.size
    if n == 0:
        return {"p": float("nan"), "n_perms": 0, "exact": False, "observed_mean": float("nan")}
    obs = float(d.mean())
    if n <= 20:
        signs = np.array(list(itertools.product([1.0, -1.0], repeat=n)))
        exact = True
    else:
        rng = np.random.default_rng(0)
        signs = rng.choice([1.0, -1.0], size=(50000, n))
        exact = False
    null = (signs * d).mean(axis=1)
    p = float(np.sum(np.abs(null) >= abs(obs) - 1e-15) / null.shape[0])
    return {"p": p, "n_perms": int(null.shape[0]), "exact": exact, "observed_mean": obs}


def subject_gaps(maps: dict, paired_subjects: list[str], bases: dict,
                 grid: list[int], conds: tuple) -> tuple[list[float], list[float], dict]:
    """Per-subject AUC(geometric)-AUC(connectome) gap for each condition."""
    per = {cond: {name: [] for name in bases} for cond in conds}
    for s in paired_subjects:
        for cond in conds:
            for name, basis in bases.items():
                per[cond][name].append(
                    auc(grid, reconstruction_curve(basis, maps[s][cond], grid))
                )
    gap_lsd = [g - c for g, c in zip(per["lsd"]["geometric"], per["lsd"]["connectome"])]
    gap_plcb = [g - c for g, c in zip(per["placebo"]["geometric"], per["placebo"]["connectome"])]
    return gap_lsd, gap_plcb, per


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modes", type=int, default=200, help="Max modes per basis")
    parser.add_argument("--edr-length", type=float, default=20.0, help="EDR decay length (mm)")
    parser.add_argument("--max-subjects", type=int, default=None)
    parser.add_argument("--subjects", nargs="*")
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()

    if args.subjects:
        subjects = list(args.subjects)
    else:
        subjects = [s for s in ALL_SUBJECTS if s not in EXCLUDE_SUBJECTS]
        if args.max_subjects:
            subjects = subjects[: args.max_subjects]

    n_modes = args.modes
    print(f"Building bases (M={n_modes}) …")
    bases = {
        "connectome": connectome_basis(n_modes),
        "geometric": geometric_basis(n_modes),
        "edr": edr_basis(n_modes, args.edr_length),
    }

    grid = build_grid(n_modes)
    conds = ("lsd", "placebo")

    print(f"Extracting per-subject parcel BOLD for {len(subjects)} subjects …")
    masker = fetch_schaefer_masker()
    # Extract raw runs once, then build raw + nuisance-regressed maps from them.
    raw_maps: dict[str, dict[str, np.ndarray | None]] = {}
    dn_maps: dict[str, dict[str, np.ndarray | None]] = {}
    kept_fracs: list[float] = []
    for sub in subjects:
        raw_maps[sub], dn_maps[sub] = {}, {}
        for cond in conds:
            try:
                runs = subject_runs(sub, cond, masker)
            except Exception as exc:  # noqa: BLE001 - skip a missing/broken run
                print(f"  ! {sub} {cond}: {exc}")
                runs = []
            raw_maps[sub][cond], _ = assemble_maps(runs, denoise=False)
            dn_maps[sub][cond], frac = assemble_maps(runs, denoise=True)
            if runs:
                kept_fracs.append(frac)

    paired_subjects = [
        s for s in subjects
        if raw_maps[s]["lsd"] is not None and raw_maps[s]["placebo"] is not None
    ]

    # Pooled per condition (drives the headline curves / chart) — raw metric.
    Y = {
        cond: np.concatenate(
            [raw_maps[s][cond] for s in subjects if raw_maps[s][cond] is not None],
            axis=1,
        )
        for cond in conds
    }

    curves = {cond: {} for cond in conds}
    for cond in conds:
        for name, basis in bases.items():
            curves[cond][name] = reconstruction_curve(basis, Y[cond], grid)
            print(f"  {cond:8s} {name:11s} acc@{grid[-1]}modes={curves[cond][name][-1]:.3f}")

    summary = {"auc": {}, "modes_to_0.5": {}, "ranking": {}}
    for cond in conds:
        summary["auc"][cond] = {k: auc(grid, curves[cond][k]) for k in bases}
        summary["modes_to_0.5"][cond] = {
            k: modes_to_threshold(grid, curves[cond][k], 0.5) for k in bases
        }
        summary["ranking"][cond] = sorted(
            bases, key=lambda k: summary["auc"][cond][k], reverse=True
        )

    geo_gap = {
        cond: summary["auc"][cond]["geometric"] - summary["auc"][cond]["connectome"]
        for cond in conds
    }
    summary["geometric_minus_connectome_auc"] = {
        **geo_gap,
        "delta_lsd_minus_placebo": geo_gap["lsd"] - geo_gap["placebo"],
    }

    # ---- Per-subject paired test (raw) + exact sign-flip permutation null -----
    print(f"Per-subject paired test ({len(paired_subjects)} subjects) …")
    gap_lsd, gap_plcb, _ = subject_gaps(raw_maps, paired_subjects, bases, grid, conds)
    delta_raw = [a - b for a, b in zip(gap_lsd, gap_plcb)]
    zeros = [0.0] * len(paired_subjects)
    shift_raw = paired_stats(gap_lsd, gap_plcb)
    shift_raw["perm"] = signflip_perm(delta_raw)
    summary["paired"] = {
        "metric": "per-subject AUC(geometric) - AUC(connectome)",
        "n_subjects": len(paired_subjects),
        "subjects": paired_subjects,
        "per_subject": {"gap_lsd": gap_lsd, "gap_placebo": gap_plcb},
        # Does the geometric advantage change between states? (the headline)
        "lsd_minus_placebo_shift": shift_raw,
        # Within each state, does geometric differ from connectome? (gap vs 0)
        "geometric_vs_connectome_lsd": paired_stats(gap_lsd, zeros),
        "geometric_vs_connectome_placebo": paired_stats(gap_plcb, zeros),
    }

    # ---- Robustness: rerun the whole paired test after nuisance regression ----
    print("Robustness: nuisance-regressed pipeline (GSR + detrend + DVARS censor) …")
    gl_dn, gp_dn, _ = subject_gaps(dn_maps, paired_subjects, bases, grid, conds)
    delta_dn = [a - b for a, b in zip(gl_dn, gp_dn)]
    shift_dn = paired_stats(gl_dn, gp_dn)
    shift_dn["perm"] = signflip_perm(delta_dn)
    survives = bool(
        shift_dn["p_ttest"] < 0.05
        and shift_dn["perm"]["p"] < 0.05
        and np.sign(shift_dn["mean_diff"]) == np.sign(shift_raw["mean_diff"])
    )
    summary["robustness"] = {
        "denoise": (
            "detrend(linear+quadratic) + global-signal regression (+ derivative) "
            "+ DVARS Tukey-fence frame censoring"
        ),
        "note": (
            "Raw OpenNeuro ds003059 has no fmriprep motion params; this is the "
            "strongest nuisance model computable from parcel time series alone. "
            "GSR removes the dominant motion/arousal confound for drug-state "
            "comparisons. Permutation = exact within-subject sign-flip null."
        ),
        "mean_frames_kept_frac": float(np.mean(kept_fracs)) if kept_fracs else float("nan"),
        "raw_shift": shift_raw,
        "denoised_shift": shift_dn,
        "denoised_per_subject": {"gap_lsd": gl_dn, "gap_placebo": gp_dn},
        "denoised_geometric_vs_connectome_lsd_p": paired_stats(gl_dn, zeros)["p_ttest"],
        "denoised_geometric_vs_connectome_placebo_p": paired_stats(gp_dn, zeros)["p_ttest"],
        "survives_nuisance_regression": survives,
    }

    summary["interpretation"] = (
        "Higher = better reconstruction with fewer modes. 'ranking' lists bases "
        "best→worst by area under the accuracy curve. 'paired' is the within-subject "
        "test of the LSD−placebo shift in AUC(geometric)−AUC(connectome) (paired t + "
        "Wilcoxon + exact sign-flip permutation, Cohen's d_z, 95% CI). 'robustness' "
        "reruns that same test after nuisance regression (GSR + detrend + DVARS "
        "censoring); 'survives_nuisance_regression' is the honest bottom line — the "
        "effect is only believable if it holds there too."
    )

    payload = {
        "meta": {
            "dataset": "OpenNeuro ds003059",
            "n_subjects": len(subjects),
            "n_modes_max": n_modes,
            "edr_length_mm": args.edr_length,
            "surface": "fsLR-32k midthickness (NSBLab/BrainEigenmodes)",
            "method": (
                "Parcellated (Schaefer-400) reconstruction-accuracy comparison of "
                "connectome / geometric (LaPy LBO) / EDR eigenmodes on ds003059 BOLD."
            ),
            "references": {
                "pang2023": "10.1038/s41586-023-06098-1",
                "mansour2024": "10.1101/2024.04.16.589843",
            },
        },
        "n_modes_grid": grid,
        "curves": curves,
        "summary": summary,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "basis_comparison.json"
    json_path.write_text(json.dumps(payload, indent=2))
    plot_results(payload, args.output_dir / "basis_comparison.png")
    print(f"Wrote {json_path}")
    print("Ranking LSD:    ", summary["ranking"]["lsd"])
    print("Ranking placebo:", summary["ranking"]["placebo"])
    print("geo−conn gap:   ", summary["geometric_minus_connectome_auc"])
    sh = summary["paired"]["lsd_minus_placebo_shift"]
    print(
        f"PAIRED shift RAW (LSD−placebo, n={sh['n']}): "
        f"Δ={sh['mean_diff']:+.4f} 95%CI[{sh['ci95'][0]:+.4f},{sh['ci95'][1]:+.4f}] "
        f"t={sh['t']:.2f} p={sh['p_ttest']:.4f} Wilcoxon p={sh['p_wilcoxon']:.4f} "
        f"perm p={sh['perm']['p']:.4f} d_z={sh['cohen_dz']:.2f}"
    )
    rb = summary["robustness"]
    dn = rb["denoised_shift"]
    print(
        f"PAIRED shift DENOISED (GSR+detrend+censor, kept "
        f"{rb['mean_frames_kept_frac']*100:.0f}% frames): "
        f"Δ={dn['mean_diff']:+.4f} 95%CI[{dn['ci95'][0]:+.4f},{dn['ci95'][1]:+.4f}] "
        f"t={dn['t']:.2f} p={dn['p_ttest']:.4f} Wilcoxon p={dn['p_wilcoxon']:.4f} "
        f"perm p={dn['perm']['p']:.4f} d_z={dn['cohen_dz']:.2f}"
    )
    print(f"SURVIVES nuisance regression: {rb['survives_nuisance_regression']}")


def plot_results(payload: dict, out_png: Path) -> None:
    grid = payload["n_modes_grid"]
    colors = {"connectome": "#c9a0ff", "geometric": "#7CE0B0", "edr": "#ffd18e"}
    paired = payload.get("summary", {}).get("paired")

    fig = plt.figure(figsize=(15.5, 4.6))
    fig.patch.set_facecolor("#0b0e14")
    ax0 = fig.add_subplot(1, 3, 1)
    ax1 = fig.add_subplot(1, 3, 2, sharey=ax0)
    ax2 = fig.add_subplot(1, 3, 3)

    for ax, cond in zip((ax0, ax1), ("placebo", "lsd")):
        ax.set_facecolor("#0b0e14")
        for name, c in colors.items():
            ax.plot(grid, payload["curves"][cond][name], color=c, lw=2, marker="o", ms=3, label=name)
        ax.set_title(cond.upper(), color="white")
        ax.set_xlabel("number of modes")
        ax.set_xscale("log")
        ax.tick_params(colors="#9fb0c8")
        for spine in ax.spines.values():
            spine.set_color("#333")
    ax0.set_ylabel("reconstruction accuracy (r)")
    ax1.legend(facecolor="#111722", edgecolor="#333", labelcolor="white")

    # Third panel: per-subject paired shift of the geometric−connectome gap.
    ax2.set_facecolor("#0b0e14")
    for spine in ax2.spines.values():
        spine.set_color("#333")
    ax2.tick_params(colors="#9fb0c8")
    if paired and paired.get("per_subject"):
        gp = paired["per_subject"]["gap_placebo"]
        gl = paired["per_subject"]["gap_lsd"]
        sh = paired["lsd_minus_placebo_shift"]
        rb = payload.get("summary", {}).get("robustness")
        x = [0, 1]
        for a, b in zip(gp, gl):
            color = "#7CE0B0" if b > a else "#ff9a9a"
            ax2.plot(x, [a, b], color=color, lw=1, alpha=0.5,
                     marker="o", ms=3, mfc=color, mec="none")
        ax2.plot(x, [float(np.mean(gp)), float(np.mean(gl))],
                 color="white", lw=2.5, marker="o", ms=6, zorder=5, label="mean (raw)")
        title2 = ""
        if rb and rb.get("denoised_per_subject"):
            gpd = rb["denoised_per_subject"]["gap_placebo"]
            gld = rb["denoised_per_subject"]["gap_lsd"]
            ax2.plot(x, [float(np.mean(gpd)), float(np.mean(gld))],
                     color="#8ec5ff", lw=2.5, ls=(0, (4, 2)), marker="s", ms=6,
                     zorder=6, label="mean (denoised)")
            dn = rb["denoised_shift"]
            title2 = f"\ndenoised: p={dn['p_ttest']:.3f}, perm p={dn['perm']['p']:.3f}"
        ax2.axhline(0, color="#555", lw=1, ls="--")
        ax2.set_xlim(-0.35, 1.35)
        ax2.set_xticks(x)
        ax2.set_xticklabels(["placebo", "LSD"])
        ax2.set_ylabel("AUC(geometric) − AUC(connectome)")
        permp = sh.get("perm", {}).get("p", float("nan"))
        ax2.set_title(
            f"paired shift  Δ={sh['mean_diff']:+.3f}  (n={sh['n']})\n"
            f"raw: p={sh['p_ttest']:.3f}, perm p={permp:.3f}, d_z={sh['cohen_dz']:.2f}"
            f"{title2}",
            color="white", fontsize=9.5,
        )
        ax2.legend(facecolor="#111722", edgecolor="#333", labelcolor="white", loc="best", fontsize=8)
    else:
        ax2.text(0.5, 0.5, "no paired data", color="#9fb0c8",
                 ha="center", va="center", transform=ax2.transAxes)

    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)


if __name__ == "__main__":
    main()
