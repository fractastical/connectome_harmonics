"""
Replication: does the LSD "geometry gains on the connectome" shift also appear
under psilocybin?

Independent dataset: OpenNeuro ds006072 (Psilocybin precision-imaging drug trial,
Siegel et al. 2025, Scientific Data). Seven healthy adults, within-subject
crossover, each scanned on a psilocybin (PSIL, 25 mg) day and an active-placebo
day (methylphenidate, MTP, 40 mg). We use the released processed resting-state
CIFTI dtseries (fsLR-32k surface, bandpassed, no GSR), parcellate to Schaefer-400,
and run the *identical* analysis as compare_bases.py:

    per subject:  gap = AUC(geometric basis) - AUC(connectome basis)
    test:         paired (PSIL - MTP) shift across subjects
                  + exact sign-flip permutation null
                  + robustness rerun after our own GSR (these data are noGSR)

This is a strong replication design: same 5-HT2A psychedelic mechanism as LSD, a
conservative *active* placebo, and surface data already in the fsLR space of our
geometric basis (cleaner than the volumetric ds003059 pipeline). Caveats: n = 7,
parcellated, active (stimulant) placebo.

Run:
    .venv-lsd/bin/python replicate_psilocybin.py
    .venv-lsd/bin/python replicate_psilocybin.py --max-subjects 2   # quick
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from compare_bases import (
    auc,
    build_grid,
    connectome_basis,
    edr_basis,
    fetch_pang,
    geometric_basis,
    paired_stats,
    reconstruction_curve,
    regress_nuisance,
    signflip_perm,
    _zscore_run,
)

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / ".cache" / "ds006072" / "ciftis"
PARC_CACHE = ROOT / ".cache" / "ds006072" / "parc"   # small (400 x T) .npy cache
OUT = ROOT / "lsd_results"
S3 = "https://s3.amazonaws.com/openneuro.org/ds006072/NON_BIDS/ciftis"
CIFTI_SUFFIX = "rsfMRI_uout_bpss_sr_noGSR_sm4.dtseries.nii"

# Drug order per subject (README "ORDER" table): which on-drug session is which.
#   sub-N  ->  {"psil": "Drug1|Drug2", "mtp": "Drug1|Drug2"}
DRUG_ORDER = {
    1: {"mtp": "Drug1", "psil": "Drug2"},   # P1: MTP, PSIL
    2: {"psil": "Drug1", "mtp": "Drug2"},   # P2: PSIL, MTP
    3: {"mtp": "Drug1", "psil": "Drug2"},   # P3: MTP, PSIL
    4: {"mtp": "Drug1", "psil": "Drug2"},   # P4: MTP, PSIL
    5: {"psil": "Drug1", "mtp": "Drug2"},   # P5: PSIL, MTP
    6: {"mtp": "Drug1", "psil": "Drug2"},   # P6: MTP, PSIL
    7: {"psil": "Drug1", "mtp": "Drug2"},   # P7: PSIL, MTP
}
SUBJECTS = list(DRUG_ORDER)
CONDS = ("psil", "mtp")


def download_cifti(sub: int, session: str) -> Path:
    """Download one ~0.5 GB CIFTI with resume + stall-retry (curl)."""
    CACHE.mkdir(parents=True, exist_ok=True)
    name = f"sub-{sub}_{session}_{CIFTI_SUFFIX}"
    local = CACHE / name
    if local.exists() and local.stat().st_size > 1_000_000:
        return local
    url = f"{S3}/{name}"
    tmp = local.with_suffix(local.suffix + ".part")
    print(f"Downloading {name} …", flush=True)

    # Authoritative expected size (Content-Length) so we never accept a truncation.
    expected = 0
    try:
        head = subprocess.run(
            ["curl", "-fsSI", "--connect-timeout", "20", url],
            capture_output=True, text=True, timeout=60,
        )
        for ln in head.stdout.splitlines():
            if ln.lower().startswith("content-length:"):
                expected = int(ln.split(":", 1)[1].strip())
    except Exception:  # noqa: BLE001
        expected = 0

    # -C - resumes the .part; --speed-time/--speed-limit abort a stalled socket
    # (no progress at <3 kB/s for 60 s) so the retry loop can resume it.
    for attempt in range(20):
        rc = subprocess.call([
            "curl", "-fsS", "-C", "-", "--connect-timeout", "30",
            "--speed-time", "60", "--speed-limit", "3000",
            "-o", str(tmp), url,
        ])
        size = tmp.stat().st_size if tmp.exists() else 0
        done = size >= expected if expected else rc == 0
        if done:
            break
        print(f"  retry {attempt+1}: {size/1e6:.0f}/{expected/1e6:.0f} MB (rc={rc})", flush=True)

    size = tmp.stat().st_size if tmp.exists() else 0
    if expected and size < expected:
        raise RuntimeError(f"download incomplete for {name}: {size}/{expected} bytes")
    if not expected and size < 1_000_000:
        raise RuntimeError(f"download failed for {name}")
    tmp.rename(local)
    return local


def load_parcels(sub: int, cond: str, keep_cifti: bool) -> np.ndarray:
    """Cached (400, T) parcel timeseries for one subject/condition.

    Downloads the big CIFTI only if the small .npy parcel cache is missing,
    then parcellates and (unless --keep-cifti) deletes the CIFTI to save disk.
    """
    PARC_CACHE.mkdir(parents=True, exist_ok=True)
    npy = PARC_CACHE / f"sub-{sub}_{cond}.npy"
    if npy.exists():
        return np.load(npy)
    session = DRUG_ORDER[sub][cond]
    path = download_cifti(sub, session)
    ts = parcellate_cifti(path)
    np.save(npy, ts)
    if not keep_cifti:
        path.unlink(missing_ok=True)
    return ts


def parcellate_cifti(path: Path) -> np.ndarray:
    """fsLR CIFTI dtseries -> (400, T) Schaefer-400 parcel means."""
    import nibabel as nib

    lh = np.loadtxt(fetch_pang("fsLR_32k_Schaefer400-lh.txt")).astype(int)
    rh = np.loadtxt(fetch_pang("fsLR_32k_Schaefer400-rh.txt")).astype(int)
    img = nib.load(str(path))
    data = img.get_fdata(dtype=np.float32)            # (T, grayordinates)
    ax = img.header.get_axis(1)
    T = data.shape[0]
    parc = np.zeros((400, T))
    cnt = np.zeros(400)
    for name, sl, bm in ax.iter_structures():
        if name == "CIFTI_STRUCTURE_CORTEX_LEFT":
            labv = lh[bm.vertex]
        elif name == "CIFTI_STRUCTURE_CORTEX_RIGHT":
            labv = rh[bm.vertex]
        else:
            continue
        seg = data[:, sl]
        for p in np.unique(labv):
            if p <= 0:
                continue
            m = labv == p
            parc[p - 1] += seg[:, m].sum(axis=1)
            cnt[p - 1] += m.sum()
    parc = parc / np.maximum(cnt[:, None], 1.0)
    return parc                                        # (400, T)


def build_maps(subjects, edr_length, keep_cifti):
    """Return raw_maps, dn_maps, kept_fracs for PSIL/MTP per subject."""
    raw_maps, dn_maps, kept = {}, {}, []
    for sub in subjects:
        raw_maps[sub], dn_maps[sub] = {}, {}
        for cond in CONDS:
            try:
                ts = load_parcels(sub, cond, keep_cifti)   # (400, T), cached
            except Exception as exc:  # noqa: BLE001
                print(f"  ! sub-{sub} {cond} ({DRUG_ORDER[sub][cond]}): {exc}")
                raw_maps[sub][cond] = dn_maps[sub][cond] = None
                continue
            raw_maps[sub][cond] = _zscore_run(ts)
            dn = regress_nuisance(ts)
            kept.append(dn.shape[1] / ts.shape[1])
            dn_maps[sub][cond] = _zscore_run(dn)
    return raw_maps, dn_maps, kept


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--modes", type=int, default=200)
    ap.add_argument("--edr-length", type=float, default=20.0)
    ap.add_argument("--max-subjects", type=int, default=None)
    ap.add_argument("--keep-cifti", action="store_true",
                    help="keep downloaded CIFTIs (~0.5 GB each) instead of deleting")
    ap.add_argument("--output-dir", type=Path, default=OUT)
    args = ap.parse_args()

    subjects = SUBJECTS[: args.max_subjects] if args.max_subjects else SUBJECTS
    n_modes = args.modes
    grid = build_grid(n_modes)

    print(f"Building bases (M={n_modes}) …")
    bases = {
        "connectome": connectome_basis(n_modes),
        "geometric": geometric_basis(n_modes),
        "edr": edr_basis(n_modes, args.edr_length),
    }

    print(f"Parcellating PSIL/MTP CIFTIs for {len(subjects)} subjects …")
    raw_maps, dn_maps, kept = build_maps(subjects, args.edr_length, args.keep_cifti)
    paired_subjects = [s for s in subjects
                       if raw_maps[s]["psil"] is not None and raw_maps[s]["mtp"] is not None]
    print(f"  usable paired subjects: {len(paired_subjects)}")

    # Pooled curves per condition (chart + ranking).
    Y = {c: np.concatenate([raw_maps[s][c] for s in paired_subjects], axis=1) for c in CONDS}
    curves = {c: {k: reconstruction_curve(b, Y[c], grid) for k, b in bases.items()} for c in CONDS}
    summary = {"auc": {}, "ranking": {}}
    for c in CONDS:
        summary["auc"][c] = {k: auc(grid, curves[c][k]) for k in bases}
        summary["ranking"][c] = sorted(bases, key=lambda k: summary["auc"][c][k], reverse=True)
        print(f"  {c:5s} AUC: " + ", ".join(f"{k}={summary['auc'][c][k]:.3f}" for k in bases))

    # Per-subject paired test (raw): PSIL - MTP shift of geometric-connectome gap.
    gap_psil = [a["geometric"] - a["connectome"] for a in _per_subject_auc(raw_maps, paired_subjects, bases, grid, "psil")]
    gap_mtp = [a["geometric"] - a["connectome"] for a in _per_subject_auc(raw_maps, paired_subjects, bases, grid, "mtp")]
    delta = [a - b for a, b in zip(gap_psil, gap_mtp)]
    zeros = [0.0] * len(paired_subjects)
    shift_raw = paired_stats(gap_psil, gap_mtp)
    shift_raw["perm"] = signflip_perm(delta)

    # Robustness: rerun after our own GSR (+detrend+censor); data are noGSR.
    gap_psil_dn = [a["geometric"] - a["connectome"] for a in _per_subject_auc(dn_maps, paired_subjects, bases, grid, "psil")]
    gap_mtp_dn = [a["geometric"] - a["connectome"] for a in _per_subject_auc(dn_maps, paired_subjects, bases, grid, "mtp")]
    delta_dn = [a - b for a, b in zip(gap_psil_dn, gap_mtp_dn)]
    shift_dn = paired_stats(gap_psil_dn, gap_mtp_dn)
    shift_dn["perm"] = signflip_perm(delta_dn)

    survives = bool(shift_dn["p_ttest"] < 0.05 and shift_dn["perm"]["p"] < 0.05
                    and np.sign(shift_dn["mean_diff"]) == np.sign(shift_raw["mean_diff"]))
    replicates = bool(shift_raw["p_ttest"] < 0.05 and shift_raw["mean_diff"] > 0)

    summary["paired"] = {
        "metric": "per-subject AUC(geometric) - AUC(connectome), PSIL - MTP",
        "n_subjects": len(paired_subjects),
        "subjects": [f"P{s}" for s in paired_subjects],
        "per_subject": {"gap_psil": gap_psil, "gap_mtp": gap_mtp},
        "psil_minus_mtp_shift": shift_raw,
        "geometric_vs_connectome_psil": paired_stats(gap_psil, zeros),
        "geometric_vs_connectome_mtp": paired_stats(gap_mtp, zeros),
    }
    summary["robustness"] = {
        "denoise": "detrend + global-signal regression (+deriv) + DVARS censoring (data are bandpassed, noGSR)",
        "mean_frames_kept_frac": float(np.mean(kept)) if kept else float("nan"),
        "denoised_shift": shift_dn,
        "denoised_per_subject": {"gap_psil": gap_psil_dn, "gap_mtp": gap_mtp_dn},
        "survives_nuisance_regression": survives,
    }
    summary["replicates_lsd_direction"] = replicates
    summary["interpretation"] = (
        "Replication of the ds003059 LSD result on an independent psilocybin cohort "
        "(ds006072). Positive PSIL-MTP shift = the geometric (shape) basis gains on "
        "the connectome (wiring) basis under psilocybin, mirroring LSD. 'survives_"
        "nuisance_regression' reruns the test after our own GSR."
    )

    payload = {
        "meta": {
            "dataset": "OpenNeuro ds006072 (psilocybin precision imaging)",
            "reference": "Siegel et al. 2025, Sci Data 10.1038/s41597-025-05189-0",
            "drug": "psilocybin 25 mg vs active placebo methylphenidate 40 mg",
            "space": "fsLR-32k surface CIFTI (bandpassed, noGSR), parcellated to Schaefer-400",
            "n_subjects": len(paired_subjects),
            "n_modes_max": n_modes,
            "edr_length_mm": args.edr_length,
            "replicates_dataset": "ds003059 (LSD)",
        },
        "n_modes_grid": grid,
        "curves": curves,
        "summary": summary,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_json = args.output_dir / "psilocybin_replication.json"
    out_json.write_text(json.dumps(payload, indent=2))
    plot_results(payload, args.output_dir / "psilocybin_replication.png")

    print(f"\nWrote {out_json}")
    sr = shift_raw
    print(f"PSIL-MTP shift RAW (n={sr['n']}): Δ={sr['mean_diff']:+.4f} "
          f"95%CI[{sr['ci95'][0]:+.4f},{sr['ci95'][1]:+.4f}] t={sr['t']:.2f} "
          f"p={sr['p_ttest']:.4f} Wilcoxon p={sr['p_wilcoxon']:.4f} "
          f"perm p={sr['perm']['p']:.4f} d_z={sr['cohen_dz']:.2f}")
    print(f"PSIL-MTP shift DENOISED: Δ={shift_dn['mean_diff']:+.4f} "
          f"p={shift_dn['p_ttest']:.4f} perm p={shift_dn['perm']['p']:.4f} "
          f"d_z={shift_dn['cohen_dz']:.2f}")
    print(f"REPLICATES LSD direction (Δ>0, p<.05): {replicates}; "
          f"survives GSR: {survives}")


def _per_subject_auc(maps, subjects, bases, grid, cond):
    """List (per subject) of {basis: AUC} for one condition."""
    out = []
    for s in subjects:
        out.append({k: auc(grid, reconstruction_curve(b, maps[s][cond], grid))
                    for k, b in bases.items()})
    return out


def plot_results(payload: dict, out_png: Path) -> None:
    grid = payload["n_modes_grid"]
    colors = {"connectome": "#c9a0ff", "geometric": "#7CE0B0", "edr": "#ffd18e"}
    paired = payload["summary"]["paired"]
    rb = payload["summary"]["robustness"]
    fig = plt.figure(figsize=(15.5, 4.6))
    fig.patch.set_facecolor("#0b0e14")
    ax0 = fig.add_subplot(1, 3, 1)
    ax1 = fig.add_subplot(1, 3, 2, sharey=ax0)
    ax2 = fig.add_subplot(1, 3, 3)
    labels = {"mtp": "METHYLPHENIDATE (placebo)", "psil": "PSILOCYBIN"}
    for ax, cond in zip((ax0, ax1), ("mtp", "psil")):
        ax.set_facecolor("#0b0e14")
        for name, c in colors.items():
            ax.plot(grid, payload["curves"][cond][name], color=c, lw=2, marker="o", ms=3, label=name)
        ax.set_title(labels[cond], color="white", fontsize=10)
        ax.set_xlabel("number of modes")
        ax.set_xscale("log")
        ax.tick_params(colors="#9fb0c8")
        for sp in ax.spines.values():
            sp.set_color("#333")
    ax0.set_ylabel("reconstruction accuracy (r)")
    ax1.legend(facecolor="#111722", edgecolor="#333", labelcolor="white")

    ax2.set_facecolor("#0b0e14")
    for sp in ax2.spines.values():
        sp.set_color("#333")
    ax2.tick_params(colors="#9fb0c8")
    gm = paired["per_subject"]["gap_mtp"]
    gp = paired["per_subject"]["gap_psil"]
    sh = paired["psil_minus_mtp_shift"]
    x = [0, 1]
    for a, b in zip(gm, gp):
        col = "#7CE0B0" if b > a else "#ff9a9a"
        ax2.plot(x, [a, b], color=col, lw=1, alpha=0.55, marker="o", ms=3, mfc=col, mec="none")
    ax2.plot(x, [float(np.mean(gm)), float(np.mean(gp))], color="white", lw=2.5,
             marker="o", ms=6, zorder=5, label="mean (raw)")
    gpd = rb["denoised_per_subject"]["gap_psil"]
    gmd = rb["denoised_per_subject"]["gap_mtp"]
    ax2.plot(x, [float(np.mean(gmd)), float(np.mean(gpd))], color="#8ec5ff", lw=2.5,
             ls=(0, (4, 2)), marker="s", ms=6, zorder=6, label="mean (denoised)")
    ax2.axhline(0, color="#555", lw=1, ls="--")
    ax2.set_xlim(-0.35, 1.35)
    ax2.set_xticks(x)
    ax2.set_xticklabels(["MTP", "PSIL"])
    ax2.set_ylabel("AUC(geometric) − AUC(connectome)")
    dn = rb["denoised_shift"]
    ax2.set_title(f"psilocybin shift  Δ={sh['mean_diff']:+.3f}  (n={sh['n']})\n"
                  f"raw: p={sh['p_ttest']:.3f}, perm p={sh['perm']['p']:.3f}, d_z={sh['cohen_dz']:.2f}\n"
                  f"denoised: p={dn['p_ttest']:.3f}, perm p={dn['perm']['p']:.3f}",
                  color="white", fontsize=9)
    ax2.legend(facecolor="#111722", edgecolor="#333", labelcolor="white", loc="best", fontsize=8)
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)


if __name__ == "__main__":
    main()
