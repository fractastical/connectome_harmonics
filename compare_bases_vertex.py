"""
Vertex-resolution Experiment 3 (geometry vs wiring) on the psilocybin cohort.

The parcellated analysis (compare_bases.py / replicate_psilocybin.py) runs at
Schaefer-400 resolution -- exactly the regime where the geometric advantage is
weakest. This script repeats the head-to-head at *vertex* resolution to test
whether the "activity shifts toward geometry under a psychedelic" effect is a
parcellation artifact or survives at high spatial resolution.

We work in fsaverage5 (10,242-vertex, left hemisphere), the resolution at which
Pang et al. 2023 (NSBLab/BrainEigenmodes) publish all three eigenbases:

    geometric  : Laplace-Beltrami eigenmodes of the fsaverage5 LH surface (LaPy)
    connectome : synthetic structural-connectome eigenmodes (Pang 2023)
    edr        : exponential-distance-rule connectome eigenmodes (Pang 2023)

The ds006072 BOLD is fsLR-32k surface CIFTI, so we resample it fsLR-32k ->
fsaverage5-10k with a pure-Python area-average resampler built from the
registration-fusion spheres bundled by neuromaps (no Connectome Workbench
needed; validated against native LaPy modes, |r|>0.95 for the lowest modes).

Honest caveats (vs the parcellated analysis):
  * Left hemisphere only (Pang's public connectome/EDR modes are LH).
  * The connectome basis is Pang's *synthetic* (model) connectome, not subject
    tractography -- but this is exactly the object of the Pang/Mansour debate.
  * 50 modes (the public connectome/EDR files stop at 50); matched across bases.
  * Psilocybin cohort only (ds003059 LSD is volumetric, no surface version).

Run:
    .venv-lsd/bin/python compare_bases_vertex.py
    .venv-lsd/bin/python compare_bases_vertex.py --max-subjects 1   # smoke test
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
CIFTI_CACHE = ROOT / ".cache" / "ds006072" / "ciftis"
VTX_CACHE = ROOT / ".cache" / "ds006072" / "vtx10k"
RESAMP = ROOT / ".cache" / "ds006072" / "fslr32k_to_fsavg10k_L.npz"
NM = Path.home() / "neuromaps-data" / "atlases"
OUT = ROOT / "lsd_results"

N_FSLR32K = 32492          # fsLR-32k vertices per hemisphere (incl. medial wall)


# ---------------------------------------------------------------------------
# fsLR-32k -> fsaverage5-10k area-average resampler (pure Python)
# ---------------------------------------------------------------------------
def build_resampler():
    """(10242 x 32492) sparse downsampling matrix, cached to disk."""
    import nibabel as nib
    from scipy.sparse import csr_matrix, save_npz, load_npz
    from scipy.spatial import cKDTree

    if RESAMP.exists():
        return load_npz(RESAMP)
    src = nib.load(
        str(NM / "fsLR" / "tpl-fsLR_space-fsaverage_den-32k_hemi-L_sphere.surf.gii")
    ).agg_data()[0]
    trg = nib.load(
        str(NM / "fsaverage" / "tpl-fsaverage_den-10k_hemi-L_sphere.surf.gii")
    ).agg_data()[0]
    ntrg, nsrc = trg.shape[0], src.shape[0]
    _, assign = cKDTree(trg).query(src, k=1)              # each src -> nearest trg
    counts = np.bincount(assign, minlength=ntrg)
    R = csr_matrix((np.ones(nsrc), (assign, np.arange(nsrc))), shape=(ntrg, nsrc))
    inv = np.zeros(ntrg)
    inv[counts > 0] = 1.0 / counts[counts > 0]
    R = csr_matrix(R.multiply(inv[:, None]))
    empty = np.where(counts == 0)[0]
    if empty.size:                                        # rare: nearest-src fallback
        _, si = cKDTree(src).query(trg[empty], k=1)
        R = R.tolil()
        for e, s in zip(empty, si):
            R[e, s] = 1.0
        R = R.tocsr()
    RESAMP.parent.mkdir(parents=True, exist_ok=True)
    save_npz(RESAMP, R)
    return R


def cortex_mask() -> np.ndarray:
    m = np.loadtxt(PANG / "fsaverage5_10k_cortex-lh_mask.txt").astype(int)
    return m > 0                                          # (10242,) bool


# ---------------------------------------------------------------------------
# Bases at fsaverage5-10k (cortex-masked, ordered low->high spatial frequency)
# ---------------------------------------------------------------------------
def geometric_basis_10k(n_modes: int, mask: np.ndarray) -> np.ndarray:
    from lapy import Solver, TriaMesh

    mesh = TriaMesh.read_vtk(str(PANG / "fsaverage5_10k_midthickness-lh.vtk"))
    fem = Solver(mesh)
    _, evecs = fem.eigs(k=n_modes + 1)
    evecs = np.asarray(evecs)[:, 1:]                      # drop constant mode
    return evecs[mask]                                    # (n_cortex, n_modes)


def _mat_modes(fname: str, n_modes: int, mask: np.ndarray) -> np.ndarray:
    import h5py

    with h5py.File(str(PANG / fname), "r") as h:
        evec = np.array(h["eig_vec"]).T                  # (10242, 50)
        eval_ = np.array(h["eig_val"]).ravel()
    order = np.argsort(eval_)                             # low -> high frequency
    evec = evec[:, order][:, :n_modes]
    return evec[mask]


def build_bases(n_modes: int, mask: np.ndarray) -> dict:
    return {
        "geometric": geometric_basis_10k(n_modes, mask),
        "connectome": _mat_modes("synthetic_connectome_eigenmodes-lh_50.mat", n_modes, mask),
        "edr": _mat_modes("synthetic_EDRconnectome_eigenmodes-lh_50.mat", n_modes, mask),
    }


# ---------------------------------------------------------------------------
# Vertex BOLD: CIFTI (fsLR-32k LH cortex) -> fsaverage5-10k cortex (cached)
# ---------------------------------------------------------------------------
def load_vertex(sub: int, cond: str, R, mask: np.ndarray, keep_cifti: bool) -> np.ndarray:
    import nibabel as nib

    VTX_CACHE.mkdir(parents=True, exist_ok=True)
    npy = VTX_CACHE / f"sub-{sub}_{cond}.npy"
    if npy.exists():
        return np.load(npy)
    session = DRUG_ORDER[sub][cond]
    path = download_cifti(sub, session)
    img = nib.load(str(path))
    data = img.get_fdata(dtype=np.float32)               # (T, grayordinates)
    ax = img.header.get_axis(1)
    T = data.shape[0]
    grid = np.zeros((T, N_FSLR32K), dtype=np.float32)     # scatter LH cortex -> 32k grid
    found = False
    for name, sl, bm in ax.iter_structures():
        if name == "CIFTI_STRUCTURE_CORTEX_LEFT":
            grid[:, bm.vertex] = data[:, sl]
            found = True
            break
    if not found:
        raise RuntimeError(f"no CORTEX_LEFT in {path.name}")
    bold10k = (R @ grid.T).astype(np.float32)            # (10242, T)
    bold = bold10k[mask]                                  # (n_cortex, T)
    np.save(npy, bold)
    if not keep_cifti:
        path.unlink(missing_ok=True)
    return bold


def build_maps(subjects, R, mask, keep_cifti):
    raw, dn, kept = {}, {}, []
    for sub in subjects:
        raw[sub], dn[sub] = {}, {}
        for cond in CONDS:
            try:
                ts = load_vertex(sub, cond, R, mask, keep_cifti)
            except Exception as exc:  # noqa: BLE001
                print(f"  ! sub-{sub} {cond} ({DRUG_ORDER[sub][cond]}): {exc}")
                raw[sub][cond] = dn[sub][cond] = None
                continue
            raw[sub][cond] = _zscore_run(ts)
            d = regress_nuisance(ts)
            kept.append(d.shape[1] / ts.shape[1])
            dn[sub][cond] = _zscore_run(d)
    return raw, dn, kept


# ---------------------------------------------------------------------------
# NOTE on the basis contrast: the public connectome ("wiring") eigenmodes are
# only Pang's *synthetic* demo surrogate (degenerate spectrum, spatially
# incoherent, reconstructs ~0), and the real tractography connectome modes are
# not published at vertex resolution. So at vertex level the only legitimate
# contrast is geometry (shape) vs EDR (distance-only connectome surrogate). The
# geometry-vs-real-wiring shift therefore stays a parcellated result; here we
# test whether the geometric basis's advantage survives off the parcellation.
GAP_REF = "edr"


def per_subject_aucs(maps, subjects, bases, grid):
    """{cond: {basis: [auc per subject]}} plus per-subject geometric-EDR gaps."""
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
    ap.add_argument("--modes", type=int, default=50)
    ap.add_argument("--max-subjects", type=int, default=None)
    ap.add_argument("--keep-cifti", action="store_true")
    args = ap.parse_args()

    OUT.mkdir(exist_ok=True)
    subjects = SUBJECTS if args.max_subjects is None else SUBJECTS[: args.max_subjects]
    grid = build_grid(args.modes)
    mask = cortex_mask()
    print(f"fsaverage5 LH cortex vertices: {int(mask.sum())} / {mask.size}")

    print("Building fsLR-32k -> fsaverage5-10k resampler …", flush=True)
    R = build_resampler()
    print("Building vertex bases (M=%d) …" % args.modes, flush=True)
    bases = build_bases(args.modes, mask)
    for b, B in bases.items():
        print(f"  {b}: {B.shape}")

    print(f"Loading/resampling vertex BOLD for {len(subjects)} subjects …", flush=True)
    raw, dn, kept = build_maps(subjects, R, mask, args.keep_cifti)

    paired = [s for s in subjects if raw[s]["psil"] is not None and raw[s]["mtp"] is not None]
    print(f"  usable paired subjects: {len(paired)}")

    raw_aucs, raw_gaps = per_subject_aucs(raw, paired, bases, grid)
    dn_aucs, dn_gaps = per_subject_aucs(dn, paired, bases, grid)
    curves = group_curves(raw, paired, bases, grid)

    mean_auc = {c: {b: float(np.mean(raw_aucs[c][b])) for b in bases} for c in CONDS}
    for c in CONDS:
        print(f"  {c:5s} AUC: " + ", ".join(f"{b}={mean_auc[c][b]:.3f}" for b in bases))

    raw_shift = paired_stats(raw_gaps["psil"], raw_gaps["mtp"])
    raw_shift["perm"] = signflip_perm(
        [a - b for a, b in zip(raw_gaps["psil"], raw_gaps["mtp"])]
    )
    dn_shift = paired_stats(dn_gaps["psil"], dn_gaps["mtp"])
    dn_shift["perm"] = signflip_perm(
        [a - b for a, b in zip(dn_gaps["psil"], dn_gaps["mtp"])]
    )

    result = {
        "meta": {
            "dataset": "OpenNeuro ds006072 (psilocybin precision imaging)",
            "analysis": "VERTEX-resolution Experiment 3 (geometry vs wiring)",
            "space": "fsaverage5 (10,242-vtx LH); ds006072 fsLR-32k CIFTI resampled "
                     "to fsaverage5 via registration-fusion spheres (area-average)",
            "bases": "geometric (LaPy LBO), connectome & EDR (Pang 2023 synthetic, "
                     "NSBLab/BrainEigenmodes)",
            "n_subjects": len(paired),
            "n_modes": args.modes,
            "hemisphere": "left",
            "cortex_vertices": int(mask.sum()),
            "connectome_note": "Public connectome modes are Pang's SYNTHETIC demo "
                               "surrogate (degenerate spectrum, ~0 reconstruction) and "
                               "real tractography connectome modes are unavailable at "
                               "vertex resolution; the legitimate vertex contrast is "
                               "geometry vs EDR. Geometry-vs-wiring stays parcellated.",
        },
        "n_modes_grid": grid,
        "curves": curves,
        "summary": {
            "auc": mean_auc,
            "paired": {
                "metric": "per-subject AUC(geometric) - AUC(edr), PSIL - MTP",
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
    (OUT / "vertex_basis_comparison.json").write_text(json.dumps(result, indent=2))
    print("\nWrote", OUT / "vertex_basis_comparison.json")

    rs, ds = raw_shift, dn_shift
    print(
        f"VERTEX PSIL-MTP shift RAW (n={rs['n']}): Δ={rs['mean_diff']:+.4f} "
        f"t={rs['t']:.2f} p={rs['p_ttest']:.4f} Wilcoxon p={rs['p_wilcoxon']:.4f} "
        f"perm p={rs['perm']['p']:.4f} d_z={rs['cohen_dz']:.2f}"
    )
    print(
        f"VERTEX PSIL-MTP shift DENOISED: Δ={ds['mean_diff']:+.4f} "
        f"p={ds['p_ttest']:.4f} perm p={ds['perm']['p']:.4f} d_z={ds['cohen_dz']:.2f}"
    )
    print(
        "REPLICATES at vertex resolution (Δ>0, perm p<.05):",
        result["summary"]["replicates_parcellated_direction"],
        "; survives GSR:",
        result["summary"]["robustness"]["survives_nuisance_regression"],
    )

    plot_results(result, raw_gaps, dn_gaps, paired)


def plot_results(result, raw_gaps, dn_gaps, paired):
    grid = result["n_modes_grid"]
    curves = result["curves"]
    colors = {"connectome": "#c9a0ff", "geometric": "#7CE0B0", "edr": "#ffd18e"}
    labels = {"connectome": "connectome (synthetic surrogate)",
              "geometric": "geometric", "edr": "edr"}
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
    rg = [a - b for a, b in zip(raw_gaps["psil"], raw_gaps["mtp"])]
    dg = [a - b for a, b in zip(dn_gaps["psil"], dn_gaps["mtp"])]
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
    ax.set_ylabel("AUC(geometric) − AUC(edr)")
    rs, ds = result["summary"]["paired"]["psil_minus_mtp_shift"], \
        result["summary"]["robustness"]["denoised_shift"]
    ax.set_title(
        f"vertex (fsaverage5-10k) shift Δ={rs['mean_diff']:+.4f} (n={rs['n']})\n"
        f"raw: p={rs['p_ttest']:.3f}, perm p={rs['perm']['p']:.3f}, d_z={rs['cohen_dz']:.2f}\n"
        f"denoised: p={ds['p_ttest']:.3f}, perm p={ds['perm']['p']:.3f}"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "vertex_basis_comparison.png", dpi=130)
    print("Wrote", OUT / "vertex_basis_comparison.png")


if __name__ == "__main__":
    main()
