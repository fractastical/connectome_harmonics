"""
Build connectome_harmonics_data.json from a real parcellated connectome.

Default source (option 1): HCP-YA group-average Schaefer-400 structural
connectivity from GNN_SC_FC (PeiyuChen2023).

Option 2: pass --source mica --sc-path /path/to/matrix.npy for a local
MICA-MICs (or other) structural connectome matrix.

Run:
    python build_connectome_data.py
    python build_connectome_data.py --source mica --sc-path ./my_sc.npy
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import urllib.request
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / ".cache"

GNN_SCFC_URL = (
    "https://github.com/PeiyuChen2023/GNN_SC_FC/raw/main/data/"
    "SC_FC_PredFC_matrix/HCPA_mean_SC_FC_predFC.mat"
)
NETNEUROLAB_SC_URL = (
    "https://github.com/netneurolab/liu_meg-scfc/raw/main/data/"
    "sc_cons_400_nosubc.npy"
)
SCHAEFER_CENTROIDS_URL = (
    "https://raw.githubusercontent.com/ThomasYeoLab/CBIG/master/"
    "stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/"
    "Parcellations/MNI/Centroid_coordinates/"
    "Schaefer2018_400Parcels_7Networks_order_FSLMNI152_2mm.Centroid_RAS.csv"
)


def download(url: str, name: str) -> Path:
    CACHE.mkdir(exist_ok=True)
    path = CACHE / name
    if not path.exists():
        print(f"Downloading {name} ...")
        data = urllib.request.urlopen(url, timeout=120).read()
        path.write_bytes(data)
    return path


def load_sc_matrix(source: str, sc_path: Path | None) -> np.ndarray:
    if source == "gnn_scfc":
        mat = loadmat(download(GNN_SCFC_URL, "HCPA_mean_SC_FC_predFC.mat"))
        w = np.asarray(mat["HCPA_mean_SC"], dtype=np.float64)
    elif source == "netneurolab":
        w = np.load(download(NETNEUROLAB_SC_URL, "sc_cons_400_nosubc.npy"))
    elif source == "mica":
        if sc_path is None:
            raise SystemExit("--sc-path is required when --source mica")
        if sc_path.suffix == ".npy":
            w = np.load(sc_path)
        else:
            w = np.loadtxt(sc_path)
    else:
        raise SystemExit(f"Unknown source: {source}")

    w = np.asarray(w, dtype=np.float64)
    if w.ndim != 2 or w.shape[0] != w.shape[1]:
        raise SystemExit(f"Expected square SC matrix, got shape {w.shape}")

    n = w.shape[0]
    if n != 400:
        print(f"Warning: expected 400 Schaefer parcels, got {n}. Using first 400 if larger.")
        if n > 400:
            w = w[:400, :400]
        else:
            raise SystemExit("Matrix smaller than 400; need Schaefer-400 or compatible atlas.")

    w = 0.5 * (w + w.T)
    np.fill_diagonal(w, 0.0)
    w[w < 0] = 0.0
    return w


def load_schaefer_centroids() -> tuple[np.ndarray, list[str], list[str]]:
    """Return (coords RAS, Yeo-7 network per parcel, hemisphere per parcel)."""
    path = download(SCHAEFER_CENTROIDS_URL, "Schaefer2018_400_centroids.csv")
    coords, networks, hemis = [], [], []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            coords.append([float(row["R"]), float(row["A"]), float(row["S"])])
            # ROI Name looks like "7Networks_LH_Vis_1"
            parts = row["ROI Name"].split("_")
            hemis.append(parts[1] if len(parts) > 1 else "NA")
            networks.append(parts[2] if len(parts) > 2 else "NA")
    coords = np.asarray(coords, dtype=np.float64)
    if coords.shape != (400, 3):
        raise SystemExit(f"Expected 400 centroid rows, got {coords.shape[0]}")
    return coords, networks, hemis


def project_nodes(coords_ras: np.ndarray) -> np.ndarray:
    """Map MNI RAS to 2D layout: x = R (L/R), y = A (A/P), normalized to ~[-1, 1]."""
    pad = 0.08

    def _norm(axis: np.ndarray) -> np.ndarray:
        scaled = (axis - axis.min()) / (axis.max() - axis.min())
        return scaled * (2 - 2 * pad) + (-1 + pad)

    return np.column_stack([_norm(coords_ras[:, 0]), _norm(coords_ras[:, 1])])


def sparsify_edges(w: np.ndarray, edge_percentile: float) -> list[list[float]]:
    mask = np.triu(np.ones_like(w, dtype=bool), k=1)
    weights = w[mask]
    positive = weights[weights > 0]
    if positive.size == 0:
        raise SystemExit("SC matrix has no positive off-diagonal weights.")

    thr = np.percentile(positive, edge_percentile)
    edges = []
    n = w.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            if w[i, j] > thr:
                weight = float(w[i, j])
                edges.append([i, j, round(weight, 4)])
    return edges


def normalized_laplacian_modes(w: np.ndarray, n_modes: int) -> tuple[np.ndarray, np.ndarray]:
    degree = w.sum(axis=1)
    degree[degree == 0] = 1.0
    d_inv_sqrt = 1.0 / np.sqrt(degree)
    norm_w = (d_inv_sqrt[:, None] * w) * d_inv_sqrt[None, :]
    lap = np.eye(w.shape[0]) - norm_w
    lap_sparse = csr_matrix(lap)

    k = n_modes + 1
    eigvals, eigvecs = eigsh(lap_sparse, k=k, which="SM")
    order = np.argsort(eigvals)
    eigvals = eigvals[order][1 : n_modes + 1]
    eigvecs = eigvecs[:, order][:, 1 : n_modes + 1]

    for i in range(n_modes):
        v = eigvecs[:, i]
        if v.sum() < 0:
            eigvecs[:, i] = -v

    return eigvals, eigvecs


def build_payload(
    w: np.ndarray,
    nodes: np.ndarray,
    coords_ras: np.ndarray,
    networks: list[str],
    hemis: list[str],
    n_modes: int,
    edge_percentile: float,
    source: str,
) -> dict:
    edges = sparsify_edges(w, edge_percentile)
    eigvals, eigvecs = normalized_laplacian_modes(w, n_modes)

    return {
        "nodes": np.round(nodes, 4).tolist(),
        # 3D anatomical node coordinates (MNI RAS, millimetres). Consumers that
        # want a true 3D layout (e.g. the embers dot cloud) sample these instead
        # of the flattened 2D `nodes`.
        "nodes3d": np.round(coords_ras, 4).tolist(),
        "edges": edges,
        "eigvals": np.round(eigvals, 6).tolist(),
        "modes": np.round(eigvecs.T, 4).tolist(),
        "networks": networks,
        "hemi": hemis,
        "meta": {
            "node_count": int(nodes.shape[0]),
            "edge_count": len(edges),
            "mode_count": n_modes,
            "source": source,
            "atlas": "Schaefer2018_400Parcels_7Networks",
            "connectome_type": "structural",
            "laplacian": "normalized (I - D^{-1/2} W D^{-1/2})",
            "edge_percentile": edge_percentile,
            "description": (
                "HCP-YA group-average Schaefer-400 structural connectome "
                "(real diffusion-MRI tractography). Modes are eigenvectors of the "
                "normalized graph Laplacian. Group-average research data — not "
                "individual or diagnostic."
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        choices=["gnn_scfc", "netneurolab", "mica"],
        default="gnn_scfc",
        help="Connectivity source (default: gnn_scfc / option 1)",
    )
    parser.add_argument(
        "--sc-path",
        type=Path,
        help="Local SC matrix (.npy or whitespace-delimited text) for --source mica",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "connectome_harmonics_data_hcp.json",
        help="Output JSON path",
    )
    parser.add_argument("--modes", type=int, default=40, help="Number of harmonic modes")
    parser.add_argument(
        "--edge-percentile",
        type=float,
        default=95.0,
        help="Keep edges with weight above this percentile of positive weights",
    )
    args = parser.parse_args()

    w = load_sc_matrix(args.source, args.sc_path)
    coords, networks, hemis = load_schaefer_centroids()
    nodes = project_nodes(coords)

    payload = build_payload(
        w, nodes, coords, networks, hemis, args.modes, args.edge_percentile, args.source
    )
    args.output.write_text(json.dumps(payload, separators=(",", ":")))
    print(f"Wrote {args.output}")
    print(
        f"  nodes={payload['meta']['node_count']}  "
        f"edges={payload['meta']['edge_count']}  "
        f"modes={payload['meta']['mode_count']}"
    )
    print(f"  λ range: {payload['eigvals'][0]:.4f} … {payload['eigvals'][-1]:.4f}")


if __name__ == "__main__":
    main()
