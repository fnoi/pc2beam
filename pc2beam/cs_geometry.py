"""
2D helpers for catalogue cross-section fitting (ported from legacy reconstruct).
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans


def subdivide_edges(
    edges: np.ndarray,
    edge_normals: np.ndarray | None = None,
    num: int | None = None,
    lmax: float | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Split edges longer than ``lmax`` into shorter segments; duplicate normals per segment.

    Matches legacy ``tools/fitting_1.subdivide_edges``.
    """
    if edge_normals is None or lmax is None:
        return edges, edge_normals

    if num is not None:
        raise NotImplementedError("num-based subdivision is not implemented")

    edge_lengths = np.linalg.norm(edges[:, 1] - edges[:, 0], axis=1)
    needs_split = edge_lengths > lmax
    n_splits = np.ceil(edge_lengths[needs_split] / lmax).astype(int) - 1

    total_new_points = int(np.sum(n_splits))
    new_size = len(edges) + total_new_points

    edges_new = np.zeros((new_size, 2, 2), dtype=np.float64)
    normals_new = np.zeros((new_size, 2), dtype=np.float64)

    normalized_normals = edge_normals / np.linalg.norm(edge_normals, axis=1)[:, np.newaxis]

    idx = 0
    split_idx = 0
    for i in range(len(edges)):
        if not needs_split[i]:
            edges_new[idx] = edges[i]
            normals_new[idx] = normalized_normals[i]
            idx += 1
        else:
            n = int(n_splits[split_idx])
            ratios = np.linspace(0.0, 1.0, n + 2)
            pts = edges[i][0] + np.outer(ratios, (edges[i][1] - edges[i][0]))
            for j in range(len(pts) - 1):
                edges_new[idx] = np.array([pts[j], pts[j + 1]])
                normals_new[idx] = normalized_normals[i]
                idx += 1
            split_idx += 1

    if edge_normals is None:
        return edges_new
    return edges_new, normals_new


def kmeans_points_normals_2D(
    points: np.ndarray,
    point_normals: np.ndarray,
    n_representatives: int,
    *,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Cluster 2D points; return cluster centers, averaged unit normals per cluster,
    per-cluster counts (weights), and point labels.

    Weights align with cluster index ``0 .. n_representatives-1`` via ``np.bincount``.
    """
    pts = np.asarray(points, dtype=np.float64)
    nrm = np.asarray(point_normals, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("points must have shape (N, 2)")
    if nrm.shape != pts.shape:
        raise ValueError("point_normals must match points shape")
    n = pts.shape[0]
    if n_representatives <= 0:
        raise ValueError("n_representatives must be positive")
    if n_representatives > n:
        raise ValueError(
            f"n_representatives ({n_representatives}) cannot exceed number of points ({n})"
        )

    kmeans = KMeans(
        n_clusters=n_representatives,
        random_state=int(random_state),
        n_init="auto",
    )
    labels = kmeans.fit_predict(pts)
    representatives = kmeans.cluster_centers_

    weights = np.bincount(labels, minlength=n_representatives).astype(np.float64)

    rep_normals = np.zeros((n_representatives, 2), dtype=np.float64)
    for i in range(n_representatives):
        cluster_mask = labels == i
        cluster_normals = nrm[cluster_mask]
        if cluster_mask.sum() == 0:
            rep_normals[i] = np.array([1.0, 0.0], dtype=np.float64)
            continue
        mean_normal = np.mean(cluster_normals, axis=0)
        norm = np.linalg.norm(mean_normal)
        rep_normals[i] = mean_normal / norm if norm >= 1e-12 else np.array([1.0, 0.0], dtype=np.float64)

    return representatives, rep_normals, weights, labels
