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


def parcel_maps(subjects: list[str], condition: str, masker) -> np.ndarray:
    """Concatenate per-parcel z-scored BOLD frames for one condition (400 x T)."""
    cols = []
    for spec in collect_runs(subjects):
        if spec["condition"] != condition:
            continue
        path = download_openneuro(spec["key"])
        ts = extract_parcel_timeseries(path, masker)         # (400, T)
        mu = ts.mean(axis=1, keepdims=True)
        sd = ts.std(axis=1, keepdims=True)
        sd[sd < 1e-9] = 1.0
        cols.append((ts - mu) / sd)
    return np.concatenate(cols, axis=1)


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
    # Per-subject, per-condition maps (z-scored per run inside parcel_maps).
    subj_maps: dict[str, dict[str, np.ndarray | None]] = {}
    for sub in subjects:
        subj_maps[sub] = {}
        for cond in conds:
            try:
                arr = parcel_maps([sub], cond, masker)
            except Exception as exc:  # noqa: BLE001 - skip a missing/broken run
                print(f"  ! {sub} {cond}: {exc}")
                arr = None
            subj_maps[sub][cond] = arr

    paired_subjects = [
        s for s in subjects
        if subj_maps[s]["lsd"] is not None and subj_maps[s]["placebo"] is not None
    ]

    # Pooled per condition (drives the headline curves / chart) — unchanged metric.
    Y = {
        cond: np.concatenate(
            [subj_maps[s][cond] for s in subjects if subj_maps[s][cond] is not None],
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

    # ---- Per-subject paired test (the defensible version of the shift) --------
    print(f"Per-subject AUC for paired test ({len(paired_subjects)} subjects) …")
    per_auc = {cond: {name: [] for name in bases} for cond in conds}
    for s in paired_subjects:
        for cond in conds:
            for name, basis in bases.items():
                c = reconstruction_curve(basis, subj_maps[s][cond], grid)
                per_auc[cond][name].append(auc(grid, c))

    gap_lsd = [
        g - c for g, c in zip(per_auc["lsd"]["geometric"], per_auc["lsd"]["connectome"])
    ]
    gap_plcb = [
        g - c
        for g, c in zip(per_auc["placebo"]["geometric"], per_auc["placebo"]["connectome"])
    ]
    zeros = [0.0] * len(paired_subjects)
    summary["paired"] = {
        "metric": "per-subject AUC(geometric) - AUC(connectome)",
        "n_subjects": len(paired_subjects),
        "subjects": paired_subjects,
        "per_subject": {"gap_lsd": gap_lsd, "gap_placebo": gap_plcb},
        # Does the geometric advantage change between states? (the headline)
        "lsd_minus_placebo_shift": paired_stats(gap_lsd, gap_plcb),
        # Within each state, does geometric differ from connectome? (gap vs 0)
        "geometric_vs_connectome_lsd": paired_stats(gap_lsd, zeros),
        "geometric_vs_connectome_placebo": paired_stats(gap_plcb, zeros),
    }

    summary["interpretation"] = (
        "Higher = better reconstruction with fewer modes. 'ranking' lists bases "
        "best→worst by area under the accuracy curve. The pooled "
        "geometric−connectome gap is suggestive; 'paired' reports the defensible "
        "within-subject test: per subject we take AUC(geometric)−AUC(connectome) "
        "for LSD and placebo and test the LSD−placebo shift across subjects "
        "(paired t + Wilcoxon, with Cohen's d_z and 95% CI)."
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
        f"PAIRED shift (LSD−placebo, n={sh['n']}): "
        f"Δ={sh['mean_diff']:+.4f} 95%CI[{sh['ci95'][0]:+.4f},{sh['ci95'][1]:+.4f}] "
        f"t={sh['t']:.2f} p={sh['p_ttest']:.4f} "
        f"Wilcoxon p={sh['p_wilcoxon']:.4f} d_z={sh['cohen_dz']:.2f}"
    )


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
        x = [0, 1]
        for a, b in zip(gp, gl):
            color = "#7CE0B0" if b > a else "#ff9a9a"
            ax2.plot(x, [a, b], color=color, lw=1, alpha=0.55,
                     marker="o", ms=3, mfc=color, mec="none")
        ax2.plot(x, [float(np.mean(gp)), float(np.mean(gl))],
                 color="white", lw=2.5, marker="o", ms=6, zorder=5, label="mean")
        ax2.axhline(0, color="#555", lw=1, ls="--")
        ax2.set_xlim(-0.35, 1.35)
        ax2.set_xticks(x)
        ax2.set_xticklabels(["placebo", "LSD"])
        ax2.set_ylabel("AUC(geometric) − AUC(connectome)")
        p = sh["p_ttest"]
        ax2.set_title(
            f"paired shift  Δ={sh['mean_diff']:+.3f}\n"
            f"t={sh['t']:.2f}, p={p:.3f}, d_z={sh['cohen_dz']:.2f}  (n={sh['n']})",
            color="white", fontsize=10,
        )
        ax2.legend(facecolor="#111722", edgecolor="#333", labelcolor="white", loc="best")
    else:
        ax2.text(0.5, 0.5, "no paired data", color="#9fb0c8",
                 ha="center", va="center", transform=ax2.transAxes)

    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)


if __name__ == "__main__":
    main()
