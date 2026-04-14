"""
Convert HELIOS LAS outputs into a demo-compatible pc2beam TXT input.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
from omegaconf import DictConfig, OmegaConf

from pc2beam.las_io import read_las_scan


def _discover_las_files(sim_output_dir: Union[str, Path]) -> List[Path]:
    root = Path(sim_output_dir)
    if not root.is_dir():
        return []
    return sorted(root.rglob("*.las"))


def _beam_part_map_from_sidecar(sidecar: DictConfig) -> Dict[int, int]:
    """
    Map HELIOS hitObjectId (part index) to beam instance_id only.
    """
    mapping = OmegaConf.select(sidecar, "beam_instance_hit_mapping") or []
    out: Dict[int, int] = {}
    for row in mapping:
        hid = row.get("helios_part_index")
        iid = row.get("instance_id")
        if hid is None or iid is None:
            continue
        out[int(hid)] = int(iid)
    return out


def _estimate_normals(points: np.ndarray, k: int = 30) -> np.ndarray:
    if len(points) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    try:
        import open3d as o3d

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
        span = np.max(points, axis=0) - np.min(points, axis=0)
        diag = float(np.linalg.norm(span))
        # Scale neighborhood to scene size while keeping a stable lower bound.
        radius = max(diag * 0.005, 0.05)
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=k)
        )
        pcd.orient_normals_to_align_with_direction(np.array([0.0, 0.0, 1.0], dtype=np.float64))
        return np.asarray(pcd.normals, dtype=np.float32)
    except ImportError:
        k_eff = int(max(3, min(k, len(points))))
        try:
            from sklearn.neighbors import KDTree

            tree = KDTree(points.astype(np.float64))
            _, nn_idx = tree.query(points, k=k_eff)
        except ImportError:
            # Pure NumPy fallback when neither Open3D nor scikit-learn is available.
            nn_idx = np.zeros((len(points), k_eff), dtype=np.int64)
            pts64 = points.astype(np.float64, copy=False)
            for i in range(len(points)):
                d2 = np.sum((pts64 - pts64[i]) ** 2, axis=1)
                nn_idx[i] = np.argpartition(d2, kth=k_eff - 1)[:k_eff]
        normals = np.zeros((len(points), 3), dtype=np.float64)
        z_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        for i in range(len(points)):
            neigh = points[nn_idx[i]]
            centered = neigh - neigh.mean(axis=0)
            cov = centered.T @ centered
            vals, vecs = np.linalg.eigh(cov)
            n = vecs[:, int(np.argmin(vals))]
            n_norm = np.linalg.norm(n)
            if n_norm > 0:
                n = n / n_norm
            if float(np.dot(n, z_up)) < 0:
                n = -n
            normals[i] = n
        return normals.astype(np.float32)


def _labels_from_hit_object_id(
    hit_object_id: Optional[np.ndarray],
    beam_part_to_instance: Dict[int, int],
    n_points: int,
    *,
    background_label: int,
) -> np.ndarray:
    labels = np.full((n_points,), int(background_label), dtype=np.int32)
    if hit_object_id is None:
        return labels
    hit = np.asarray(hit_object_id).astype(np.int64, copy=False)
    for part_idx, inst_id in beam_part_to_instance.items():
        labels[hit == int(part_idx)] = int(inst_id)
    return labels


def export_helios_sim_to_pc2beam_txt(
    sim_output_dir: Union[str, Path],
    sidecar: DictConfig,
    output_txt_path: Union[str, Path],
    *,
    background_label: int = -1,
    normal_knn: int = 30,
) -> Dict[str, Any]:
    """
    Export HELIOS LAS results to a 7-column txt:
    ``x y z nx ny nz instance_id``.
    """
    las_files = _discover_las_files(sim_output_dir)
    if not las_files:
        raise FileNotFoundError(f"No LAS files found under {Path(sim_output_dir).resolve()}")

    beam_part_to_instance = _beam_part_map_from_sidecar(sidecar)
    point_chunks: List[np.ndarray] = []
    label_chunks: List[np.ndarray] = []

    for las_path in las_files:
        scan = read_las_scan(las_path)
        pts = np.asarray(scan.points, dtype=np.float32)
        labels = _labels_from_hit_object_id(
            scan.hit_object_id,
            beam_part_to_instance,
            len(pts),
            background_label=background_label,
        )
        point_chunks.append(pts)
        label_chunks.append(labels)

    points = np.concatenate(point_chunks, axis=0)
    labels = np.concatenate(label_chunks, axis=0)
    normals = _estimate_normals(points, k=normal_knn)

    output_txt_path = Path(output_txt_path)
    output_txt_path.parent.mkdir(parents=True, exist_ok=True)
    data = np.column_stack([points, normals, labels])
    np.savetxt(output_txt_path, data, fmt=["%.6f", "%.6f", "%.6f", "%.6f", "%.6f", "%.6f", "%d"])

    labeled_mask = labels != int(background_label)
    return {
        "output_txt": output_txt_path.resolve(),
        "las_files": [p.resolve() for p in las_files],
        "point_count": int(len(points)),
        "labeled_beam_point_count": int(np.count_nonzero(labeled_mask)),
        "background_point_count": int(np.count_nonzero(~labeled_mask)),
        "background_label": int(background_label),
    }
