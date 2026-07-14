"""
Vertex-resolution Experiment 3 (geometry vs REAL wiring) on the psilocybin cohort.

This is the upgrade the parcellated and fsaverage5 analyses could not do: a fair
geometry-vs-*wiring* head-to-head at full vertex resolution, using Pang et al.
2023's **empirical high-resolution group-average connectome** (not the synthetic
surrogate). All three objects live in the same fsLR-32k left-hemisphere cortex
(medial wall removed, 29,696 vertices):

    geometric  : Laplace-Beltrami eigenmodes of the fsLR-32k LH midthickness
                 surface (Pang 2023, template_eigenmodes), masked to cortex.
    connectome : graph-Laplacian (connectome-harmonic) eigenmodes we compute here
                 from Pang's S255 high-resolution group-average tractography
                 connectome (empirical/S255_high-resolution_group_average_
                 connectome_cortex_nomedial-lh.mat, 29696 x 29696).

The ds006072 psilocybin BOLD is already fsLR-32k surface CIFTI, so its LH cortex
grayordinates align vertex-for-vertex with both bases -- no resampling needed
(unlike the fsaverage5 analysis in compare_bases_vertex.py).

Per subject:  gap = AUC(geometric) - AUC(connectome)
Test:         paired (PSIL - MTP) shift across subjects
              + exact sign-flip permutation null
              + robustness rerun after our own GSR (these data are noGSR)

Honest caveats:
  * Left hemisphere only (Pang's empirical connectome release is LH cortex).
  * Group-average connectome (S255 HCP), not subject-specific tractography.
  * Psilocybin cohort only (ds003059 LSD is volumetric, no surface version).

Run:
    .venv-lsd/bin/python compare_bases_vertex_fslr.py
    .venv-lsd/bin/python compare_bases_vertex_fslr.py --max-subjects 1   # smoke
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from compare_bases import (
    auc,
    build_grid,
    paired_stats,
    reconstruction_curve,
    regress_nuisance,
    signflip_perm,
    _zscore_run,
)
from replicate_psilocybin import (
    CONDS,
    DRUG_ORDER,
    SUBJECTS,
    download_cifti,
)

ROOT = Path(__file__).resolve().parent
PANG = ROOT / ".cache" / "pang"
VTX_CACHE = ROOT / ".cache" / "ds006072" / "vtxfslr"
OUT = ROOT / "lsd_results"

N_FSLR32K = 32492                      # fsLR-32k vertices per hemisphere
GEOM_NPY = PANG / "geom_fslr_lh_200.npy"
GEOM_TXT = PANG / "fsLR_32k_midthickness-lh_emode_200.txt"
GEOM_EVAL = PANG / "fsLR_32k_midthickness-lh_eval_200.txt"
MASK_TXT = PANG / "fsLR_32k_cortex-lh_mask.txt"
CONN_MAT = PANG / "S255_connectome_cortex_nomedial-lh.mat"
CONN_CACHE = PANG / "connectome_fslr_lh_eigmodes.npz"

GAP_REF = "connectome"                 # the real-wiring basis (the whole point)


# ---------------------------------------------------------------------------
# fsLR-32k LH cortex mask (medial wall removed -> 29,696 vertices)
# ---------------------------------------------------------------------------
def cortex_idx() -> np.ndarray:
    m = np.loadtxt(MASK_TXT).astype(int)
    return np.where(m > 0)[0]          # ascending fsLR vertex indices


# ---------------------------------------------------------------------------
# Geometric basis: Pang's fsLR-32k LBO eigenmodes, masked to cortex, drop const
# ---------------------------------------------------------------------------
def geometric_basis(n_modes: int, idx: np.ndarray) -> np.ndarray:
    G = np.load(GEOM_NPY) if GEOM_NPY.exists() else np.loadtxt(GEOM_TXT, dtype=np.float32)
    G = G[idx]                          # (29696, 200) cortex vertices
    return G[:, 1 : n_modes + 1]        # drop mode 0 (constant); next n_modes


# ---------------------------------------------------------------------------
# Connectome harmonics: compute from Pang's empirical group-average connectome
#   L = I - D^-1/2 W D^-1/2 ; we take the lowest-frequency non-trivial modes.
#   (== largest eigenvalues of the normalized adjacency A = D^-1/2 W D^-1/2)
# ---------------------------------------------------------------------------
def _load_connectome():
    """Return W (dense float32 or scipy.sparse) of shape (29696, 29696)."""
    import scipy.sparse as sp

    try:                                # pre-7.3 .mat
        from scipy.io import loadmat

        m = loadmat(str(CONN_MAT))
        keys = [k for k in m if not k.startswith("__")]
        arr = max((m[k] for k in keys), key=lambda a: getattr(a, "size", 0))
        return arr
    except NotImplementedError:         # v7.3 (HDF5)
        import h5py

        with h5py.File(str(CONN_MAT), "r") as h:
            # MATLAB sparse -> group with data/ir/jc ; dense -> dataset
            def find(obj, depth=0):
                best = None
                for k in obj:
                    v = obj[k]
                    if isinstance(v, h5py.Group) and {"data", "ir", "jc"} <= set(v):
                        return ("sparse", v)
                    if isinstance(v, h5py.Dataset) and v.ndim == 2:
                        if best is None or v.size > best[1].size:
                            best = ("dense", v)
                    if isinstance(v, h5py.Group) and depth < 2:
                        r = find(v, depth + 1)
                        if r and r[0] == "sparse":
                            return r
                return best

            kind, v = find(h)
            if kind == "sparse":
                data = np.array(v["data"]).ravel()
                ir = np.array(v["ir"]).ravel().astype(np.int64)
                jc = np.array(v["jc"]).ravel().astype(np.int64)
                n = jc.size - 1
                W = sp.csc_matrix((data, ir, jc), shape=(n, n))
                return W
            return np.array(v, dtype=np.float32).T


def connectome_basis(n_modes: int) -> tuple[np.ndarray, np.ndarray]:
    import scipy.sparse as sp
    from scipy.sparse.linalg import eigsh, LinearOperator

    if CONN_CACHE.exists():
        z = np.load(CONN_CACHE)
        if z["evecs"].shape[1] >= n_modes:
            return z["evecs"][:, :n_modes], z["evals"][:n_modes]

    print("  loading empirical connectome (3.5 GB) …", flush=True)
    W = _load_connectome()
    sparse = sp.issparse(W)
    print(f"  connectome: shape={W.shape} sparse={sparse} "
          f"dtype={W.dtype}", flush=True)

    # symmetrize, drop self-loops, clip negatives
    if sparse:
        W = (W + W.T) * 0.5
        W = W.tolil(); W.setdiag(0.0); W = W.tocsr()
        W.data[W.data < 0] = 0.0
        W.eliminate_zeros()
        deg = np.asarray(W.sum(axis=1)).ravel()
    else:
        W = np.asarray(W, dtype=np.float32)
        W = 0.5 * (W + W.T)
        np.fill_diagonal(W, 0.0)
        W[W < 0] = 0.0
        deg = W.sum(axis=1)

    n = W.shape[0]
    dis = np.zeros(n, dtype=np.float32)
    nz = deg > 0
    dis[nz] = 1.0 / np.sqrt(deg[nz])

    # A = D^-1/2 W D^-1/2  (normalized adjacency); largest eigs -> lowest L modes
    if sparse:
        D = sp.diags(dis)
        A = (D @ W @ D).tocsr()
        evals_A, evecs = eigsh(A, k=n_modes + 1, which="LA")
    else:
        Wf = W
        def matvec(x):
            x = np.asarray(x, dtype=np.float32).ravel()
            return (dis * (Wf @ (dis * x))).astype(np.float64)
        A = LinearOperator((n, n), matvec=matvec, dtype=np.float64)
        evals_A, evecs = eigsh(A, k=n_modes + 1, which="LA")

    order = np.argsort(evals_A)[::-1]          # A large -> L small (low freq)
    evals_A = evals_A[order]
    evecs = evecs[:, order]
    lap_evals = 1.0 - evals_A                   # graph-Laplacian eigenvalues
    # drop the trivial (lowest-frequency) mode
    evecs = np.ascontiguousarray(evecs[:, 1 : n_modes + 1], dtype=np.float32)
    lap_evals = lap_evals[1 : n_modes + 1]
    for i in range(evecs.shape[1]):             # sign convention
        if evecs[:, i].sum() < 0:
            evecs[:, i] = -evecs[:, i]
    np.savez(CONN_CACHE, evecs=evecs, evals=lap_evals)
    print(f"  connectome modes: {evecs.shape}, "
          f"lap eigenvalues [{lap_evals[0]:.4f} .. {lap_evals[-1]:.4f}]", flush=True)
    return evecs, lap_evals


# ---------------------------------------------------------------------------
# Vertex BOLD: ds006072 fsLR-32k CIFTI -> LH cortex (29696, T), cached
# ---------------------------------------------------------------------------
def load_vertex(sub: int, cond: str, idx: np.ndarray, keep_cifti: bool) -> np.ndarray:
    import nibabel as nib

    VTX_CACHE.mkdir(parents=True, exist_ok=True)
    npy = VTX_CACHE / f"sub-{sub}_{cond}.npy"
    if npy.exists():
        return np.load(npy)
    session = DRUG_ORDER[sub][cond]
    path = download_cifti(sub, session)
    img = nib.load(str(path))
    data = img.get_fdata(dtype=np.float32)          # (T, grayordinates)
    ax = img.header.get_axis(1)
    T = data.shape[0]
    grid = np.zeros((T, N_FSLR32K), dtype=np.float32)
    found = False
    for name, sl, bm in ax.iter_structures():
        if name == "CIFTI_STRUCTURE_CORTEX_LEFT":
            grid[:, bm.vertex] = data[:, sl]
            found = True
            break
    if not found:
        raise RuntimeError(f"no CORTEX_LEFT in {path.name}")
    bold = grid[:, idx].T.astype(np.float32)        # (29696, T)
    np.save(npy, bold)
    if not keep_cifti:
        path.unlink(missing_ok=True)
    return bold


def build_maps(subjects, idx, keep_cifti):
    raw, dn, kept = {}, {}, []
    for sub in subjects:
        raw[sub], dn[sub] = {}, {}
        for cond in CONDS:
            try:
                ts = load_vertex(sub, cond, idx, keep_cifti)
            except Exception as exc:  # noqa: BLE001
                print(f"  ! sub-{sub} {cond} ({DRUG_ORDER[sub][cond]}): {exc}")
                raw[sub][cond] = dn[sub][cond] = None
                continue
            raw[sub][cond] = _zscore_run(ts)
            d = regress_nuisance(ts)
            kept.append(d.shape[1] / ts.shape[1])
            dn[sub][cond] = _zscore_run(d)
    return raw, dn, kept


def per_subject_aucs(maps, subjects, bases, grid):
    out = {c: {b: [] for b in bases} for c in CONDS}
    for sub in subjects:
        for cond in CONDS:
            Y = maps[sub][cond]
            for b, B in bases.items():
                out[cond][b].append(auc(grid, reconstruction_curve(B, Y, grid)))
    gaps = {
        c: [g - k for g, k in zip(out[c]["geometric"], out[c][GAP_REF])]
        for c in CONDS
    }
    return out, gaps


def group_curves(maps, subjects, bases, grid):
    curves = {c: {b: None for b in bases} for c in CONDS}
    for cond in CONDS:
        for b, B in bases.items():
            acc = [reconstruction_curve(B, maps[s][cond], grid) for s in subjects]
            curves[cond][b] = list(np.mean(acc, axis=0))
    return curves


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--modes", type=int, default=199)
    ap.add_argument("--max-subjects", type=int, default=None)
    ap.add_argument("--keep-cifti", action="store_true")
    args = ap.parse_args()

    OUT.mkdir(exist_ok=True)
    subjects = SUBJECTS if args.max_subjects is None else SUBJECTS[: args.max_subjects]
    idx = cortex_idx()
    print(f"fsLR-32k LH cortex vertices: {idx.size}")
    grid = build_grid(args.modes)

    print("Building geometric basis …", flush=True)
    geo = geometric_basis(args.modes, idx)
    print(f"  geometric: {geo.shape}")
    print("Building empirical-connectome basis …", flush=True)
    con, con_eval = connectome_basis(args.modes)
    M = min(geo.shape[1], con.shape[1])
    bases = {"geometric": geo[:, :M], "connectome": con[:, :M]}
    grid = build_grid(M)
    print(f"  using M={M} modes; grid={grid}")

    print(f"Loading vertex BOLD for {len(subjects)} subjects …", flush=True)
    raw, dn, kept = build_maps(subjects, idx, args.keep_cifti)
    paired = [s for s in subjects if raw[s]["psil"] is not None and raw[s]["mtp"] is not None]
    print(f"  usable paired subjects: {len(paired)}")

    raw_aucs, raw_gaps = per_subject_aucs(raw, paired, bases, grid)
    dn_aucs, dn_gaps = per_subject_aucs(dn, paired, bases, grid)
    curves = group_curves(raw, paired, bases, grid)

    mean_auc = {c: {b: float(np.mean(raw_aucs[c][b])) for b in bases} for c in CONDS}
    for c in CONDS:
        print(f"  {c:5s} AUC: " + ", ".join(f"{b}={mean_auc[c][b]:.3f}" for b in bases))

    raw_shift = paired_stats(raw_gaps["psil"], raw_gaps["mtp"])
    raw_shift["perm"] = signflip_perm([a - b for a, b in zip(raw_gaps["psil"], raw_gaps["mtp"])])
    dn_shift = paired_stats(dn_gaps["psil"], dn_gaps["mtp"])
    dn_shift["perm"] = signflip_perm([a - b for a, b in zip(dn_gaps["psil"], dn_gaps["mtp"])])

    result = {
        "meta": {
            "dataset": "OpenNeuro ds006072 (psilocybin precision imaging)",
            "analysis": "VERTEX-resolution Experiment 3 (geometry vs REAL wiring)",
            "space": "fsLR-32k LH cortex (29,696 vtx, medial wall removed); "
                     "ds006072 CIFTI grayordinates used natively (no resampling)",
            "bases": "geometric (Pang 2023 fsLR-32k LBO eigenmodes) vs connectome "
                     "(graph-Laplacian harmonics of Pang's S255 high-resolution "
                     "group-average empirical tractography connectome)",
            "n_subjects": len(paired),
            "n_modes": M,
            "hemisphere": "left",
            "cortex_vertices": int(idx.size),
            "connectome_note": "Empirical group-average connectome (S255 HCP), the "
                               "real-wiring basis -- not the synthetic surrogate used "
                               "in the fsaverage5 check. This is a genuine geometry-vs-"
                               "wiring contrast at vertex resolution.",
        },
        "n_modes_grid": grid,
        "curves": curves,
        "summary": {
            "auc": mean_auc,
            "paired": {
                "metric": "per-subject AUC(geometric) - AUC(connectome), PSIL - MTP",
                "n_subjects": len(paired),
                "psil_minus_mtp_shift": raw_shift,
            },
            "robustness": {
                "denoise": "detrend + GSR(+deriv) + DVARS censoring",
                "mean_frames_kept_frac": float(np.mean(kept)) if kept else None,
                "denoised_shift": dn_shift,
                "survives_nuisance_regression": bool(
                    dn_shift["mean_diff"] > 0 and dn_shift["perm"]["p"] < 0.05
                ),
            },
            "replicates_parcellated_direction": bool(
                raw_shift["mean_diff"] > 0 and raw_shift["perm"]["p"] < 0.05
            ),
        },
    }
    (OUT / "vertex_fslr_basis_comparison.json").write_text(json.dumps(result, indent=2))
    print("\nWrote", OUT / "vertex_fslr_basis_comparison.json")

    rs, ds = raw_shift, dn_shift
    print(f"VERTEX(fsLR) PSIL-MTP shift RAW (n={rs['n']}): Δ={rs['mean_diff']:+.4f} "
          f"t={rs['t']:.2f} p={rs['p_ttest']:.4f} Wilcoxon p={rs['p_wilcoxon']:.4f} "
          f"perm p={rs['perm']['p']:.4f} d_z={rs['cohen_dz']:.2f}")
    print(f"VERTEX(fsLR) PSIL-MTP shift DENOISED: Δ={ds['mean_diff']:+.4f} "
          f"p={ds['p_ttest']:.4f} perm p={ds['perm']['p']:.4f} d_z={ds['cohen_dz']:.2f}")
    print("REPLICATES at vertex resolution (Δ>0, perm p<.05):",
          result["summary"]["replicates_parcellated_direction"],
          "; survives GSR:", result["summary"]["robustness"]["survives_nuisance_regression"])

    plot_results(result, raw_gaps, dn_gaps, paired)


def plot_results(result, raw_gaps, dn_gaps, paired):
    grid = result["n_modes_grid"]
    curves = result["curves"]
    colors = {"connectome": "#c9a0ff", "geometric": "#7CE0B0"}
    labels = {"connectome": "connectome (real wiring)", "geometric": "geometric (shape)"}
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    for ax, cond, title in zip(axes[:2], ["mtp", "psil"],
                               ["METHYLPHENIDATE (placebo)", "PSILOCYBIN"]):
        for b, c in colors.items():
            ax.plot(grid, curves[cond][b], "-o", ms=3, color=c, label=labels[b])
        ax.set_title(title)
        ax.set_xlabel("number of modes")
        ax.set_ylabel("reconstruction accuracy")
        ax.set_xscale("log")
        ax.legend()

    ax = axes[2]
    for i in range(len(paired)):
        ax.plot([0, 1], [raw_gaps["mtp"][i], raw_gaps["psil"][i]],
                color="#7e8aa2", lw=1, alpha=.7)
    ax.plot([0, 1], [np.mean(raw_gaps["mtp"]), np.mean(raw_gaps["psil"])],
            "-o", color="white", lw=2.5, label="mean (raw)")
    ax.plot([0, 1], [np.mean(dn_gaps["mtp"]), np.mean(dn_gaps["psil"])],
            "--s", color="#6fb1ff", lw=2, label="mean (denoised)")
    ax.axhline(0, color="#555", lw=.8, ls=":")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["MTP", "PSIL"])
    ax.set_ylabel("AUC(geometric) − AUC(connectome)")
    rs = result["summary"]["paired"]["psil_minus_mtp_shift"]
    ds = result["summary"]["robustness"]["denoised_shift"]
    ax.set_title(
        f"vertex (fsLR-32k) geometry−wiring shift Δ={rs['mean_diff']:+.4f} (n={rs['n']})\n"
        f"raw: p={rs['p_ttest']:.3f}, perm p={rs['perm']['p']:.3f}, d_z={rs['cohen_dz']:.2f}\n"
        f"denoised: p={ds['p_ttest']:.3f}, perm p={ds['perm']['p']:.3f}"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "vertex_fslr_basis_comparison.png", dpi=130)
    print("Wrote", OUT / "vertex_fslr_basis_comparison.png")


if __name__ == "__main__":
    main()
