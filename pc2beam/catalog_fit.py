"""
Cross-section fitting from IFC catalogue (legacy-style objectives).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from .cs_geometry import subdivide_edges


def params2verts(solution: np.ndarray, from_cog: bool = True) -> np.ndarray:
    if from_cog:
        tf, tw, bf, d = solution
        x0 = -bf / 2.0
        y0 = -d / 2.0
    else:
        x0, y0, tf, tw, bf, d = solution
    v0 = np.array([x0, y0])
    v1 = v0 + np.array([0, tf])
    v2 = v1 + np.array([(bf / 2 - tw / 2), 0])
    v3 = v2 + np.array([0, (d - 2 * tf)])
    v5 = v0 + np.array([0, d])
    v4 = v5 - np.array([0, tf])
    v6 = v5 + np.array([bf, 0])
    v7 = v4 + np.array([bf, 0])
    v8 = v3 + np.array([tw, 0])
    v9 = v2 + np.array([tw, 0])
    v11 = v0 + np.array([bf, 0])
    v10 = v11 + np.array([0, tf])
    return np.array([v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11], dtype=np.float64)


def verts2edges(vertices: np.ndarray) -> np.ndarray:
    num_edges = vertices.shape[0]
    edges = np.zeros((num_edges, 2, 2), dtype=np.float64)
    for i in range(num_edges):
        edges[i] = np.array([vertices[i], vertices[(i + 1) % num_edges]], dtype=np.float64)
    return edges


def get_solution_edge_normals() -> np.ndarray:
    n_up = [0, 1]
    n_down = [0, -1]
    n_left = [-1, 0]
    n_right = [1, 0]
    return np.array(
        [n_left, n_up, n_left, n_down, n_left, n_up, n_right, n_down, n_right, n_up, n_right, n_down],
        dtype=np.float64,
    )


def point_segment_distance(points: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    v = p2 - p1
    denom = float(np.dot(v, v))
    if denom < 1e-12:
        return np.linalg.norm(points - p1, axis=1)
    t = np.maximum(0.0, np.minimum(1.0, np.dot(points - p1, v) / denom))
    projections = p1 + t[:, np.newaxis] * v
    return np.linalg.norm(points - projections, axis=1)


def _cost_combined(
    solution_params: np.ndarray,
    points: np.ndarray,
    normals: np.ndarray,
    *,
    polygon_subdivision: bool = False,
    edge_subdivision_lmax: float = 0.01,
    cluster_weights: Optional[np.ndarray] = None,
    use_cluster_weights: bool = True,
) -> tuple[float, float, float]:
    """
    Legacy-style three objectives (minimize log-distance and inactive edge fraction,
    maximize orientation cosine sum — combined with scalar ``log - coverage - cosine``).

    When ``polygon_subdivision`` is False and no cluster weights are applied, edge
    activity uses the same ``edge_attribute_no > 0`` mask as the full legacy path.
    """
    data_points = np.asarray(points, dtype=np.float64)
    data_normals = np.asarray(normals, dtype=np.float64)
    n_pts = data_points.shape[0]
    if n_pts == 0:
        raise ValueError("empty point set in cost")

    verts = params2verts(solution_params, from_cog=False)
    solution_edges = verts2edges(verts)
    solution_edge_normals = get_solution_edge_normals()

    if polygon_subdivision:
        se, sn = subdivide_edges(
            edges=solution_edges,
            edge_normals=solution_edge_normals,
            lmax=float(edge_subdivision_lmax),
        )
        solution_edges = se
        solution_edge_normals = sn

    w_sim = None
    w_dist = None
    if cluster_weights is not None and use_cluster_weights:
        cw = np.asarray(cluster_weights, dtype=np.float64).reshape(-1)
        if cw.shape[0] != n_pts:
            raise ValueError(
                f"cluster_weights length {cw.shape[0]} must match number of points {n_pts}"
            )
        w_sim = cw
        w_dist = cw

    # Pre-compute cosine similarities (rows = edges, cols = points), legacy fitting_nsga style.
    all_similarities = np.array(
        [
            cosine_similarity(normal.reshape(1, -1), data_normals)[0]
            for normal in solution_edge_normals
        ]
    )
    if w_sim is not None:
        all_similarities = all_similarities * w_sim

    edge_lengths = np.linalg.norm(solution_edges[:, 1] - solution_edges[:, 0], axis=1)
    edge_length_total = float(np.sum(edge_lengths))

    edge_distances = np.array(
        [point_segment_distance(data_points, edge[0], edge[1]) for edge in solution_edges],
        dtype=np.float64,
    )
    if w_dist is not None:
        edge_distances = edge_distances * w_dist

    best_edge_per_point = np.argmin(edge_distances, axis=0)
    min_distances_per_point = np.min(edge_distances, axis=0)
    min_distances_per_point = np.maximum(min_distances_per_point, 1e-10)

    log_distance = float(np.sum(np.log(min_distances_per_point)) / len(data_points))

    edge_activity_flag = np.zeros(len(solution_edges))
    edge_attribute_no = np.zeros(len(solution_edges))

    for edge in range(len(solution_edges)):
        if edge in best_edge_per_point:
            edge_activity_flag[edge] = 1.0
            edge_attribute_no[edge] = float(np.sum(best_edge_per_point == edge))

    ref_no = float(np.max(edge_attribute_no)) if len(edge_attribute_no) else 0.0
    edge_activity_raw = edge_activity_flag.copy()
    edge_activity = edge_attribute_no > 0.0 * ref_no

    orientation_quality = 0.0
    for edge_id in range(len(solution_edges)):
        if edge_activity_raw[edge_id]:
            activating_points = np.where(best_edge_per_point == edge_id)[0]
            orientation_quality += float(np.sum(all_similarities[edge_id, activating_points]))

    cosine_quality = orientation_quality / len(data_points)

    active_edge_length_relative = float(np.sum(edge_activity * edge_lengths) / max(edge_length_total, 1e-12))

    return log_distance, active_edge_length_relative, cosine_quality


@dataclass
class FitConfig:
    n_pop: int = 100
    n_gen: int = 20
    random_seed: int = 42
    n_elite: int = 20
    mutation_scale_xy: float = 0.03
    polygon_subdivision: bool = False
    edge_subdivision_lmax: float = 0.01
    use_cluster_weights: bool = True


def _setup_lims(points: np.ndarray) -> tuple[tuple[float, float], tuple[float, float]]:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    extent_x = abs(maxs[0] - mins[0])
    extent_y = abs(maxs[1] - mins[1])
    return (mins[0] - 0.1 * extent_x, mins[0] + 0.5 * extent_x), (mins[1] - 0.1 * extent_y, mins[1] + 0.5 * extent_y)


def solve_w_nsga_style(
    points: np.ndarray,
    normals: np.ndarray,
    catalogue_df,
    fit_cfg: Optional[FitConfig] = None,
    cluster_weights: Optional[np.ndarray] = None,
    show_progress: bool = False,
    progress_label: str = "fit",
    progress_min_interval: float = 0.25,
) -> Dict[str, Any]:
    """
    A lightweight NSGA-style evolutionary search (3 objectives).
    """
    if fit_cfg is None:
        fit_cfg = FitConfig()
    rng = np.random.default_rng(int(fit_cfg.random_seed))
    pts = np.asarray(points, dtype=np.float64)
    nrm = np.asarray(normals, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2 or nrm.shape != pts.shape:
        raise ValueError("points and normals must have shape (N, 2)")
    if len(catalogue_df) == 0:
        raise ValueError("empty catalogue")

    cw = cluster_weights
    if cw is not None and not fit_cfg.use_cluster_weights:
        cw = None

    x_range, y_range = _setup_lims(pts)
    catalogue = catalogue_df[["tw", "tf", "bf", "d"]].to_numpy(dtype=np.float64)
    n_catalogue = len(catalogue)

    pop = np.column_stack(
        [
            rng.integers(0, n_catalogue, size=fit_cfg.n_pop),
            rng.uniform(x_range[0], x_range[1], size=fit_cfg.n_pop),
            rng.uniform(y_range[0], y_range[1], size=fit_cfg.n_pop),
        ]
    )

    best = None
    best_score = None
    generations = range(int(fit_cfg.n_gen))
    if show_progress:
        generations = tqdm(
            generations,
            total=int(fit_cfg.n_gen),
            desc=f"{progress_label} generations",
            unit="gen",
            leave=False,
            mininterval=max(0.0, float(progress_min_interval)),
        )
    for _ in generations:
        scored = []
        for ind in pop:
            cidx = int(np.clip(np.round(ind[0]), 0, n_catalogue - 1))
            params = np.array([ind[1], ind[2], catalogue[cidx][1], catalogue[cidx][0], catalogue[cidx][2], catalogue[cidx][3]])
            obj = _cost_combined(
                params,
                pts,
                nrm,
                polygon_subdivision=fit_cfg.polygon_subdivision,
                edge_subdivision_lmax=fit_cfg.edge_subdivision_lmax,
                cluster_weights=cw,
                use_cluster_weights=fit_cfg.use_cluster_weights,
            )
            scalar = obj[0] - obj[1] - obj[2]
            scored.append((scalar, obj, cidx, params))
            if best_score is None or scalar < best_score:
                best_score = scalar
                best = (obj, cidx, params.copy())
        scored.sort(key=lambda x: x[0])
        elite = scored[: max(2, int(fit_cfg.n_elite))]
        parents = np.array([[e[2], e[3][0], e[3][1]] for e in elite], dtype=np.float64)
        next_pop = []
        next_pop.extend(parents.tolist())
        while len(next_pop) < int(fit_cfg.n_pop):
            p = parents[rng.integers(0, len(parents))]
            child = p.copy()
            if rng.random() < 0.4:
                child[0] = np.clip(child[0] + rng.integers(-3, 4), 0, n_catalogue - 1)
            child[1] += rng.normal(0.0, fit_cfg.mutation_scale_xy)
            child[2] += rng.normal(0.0, fit_cfg.mutation_scale_xy)
            child[1] = np.clip(child[1], x_range[0], x_range[1])
            child[2] = np.clip(child[2], y_range[0], y_range[1])
            next_pop.append(child.tolist())
        pop = np.asarray(next_pop, dtype=np.float64)
        if show_progress and best_score is not None:
            generations.set_postfix(best_score=float(best_score), refresh=False)

    assert best is not None
    best_obj, best_idx, best_params = best
    verts = params2verts(best_params, from_cog=False)
    cs_cog_x = verts[0][0] + (verts[6][0] - verts[0][0]) / 2.0
    cs_cog_y = verts[0][1] + (verts[6][1] - verts[0][1]) / 2.0
    return {
        "h_beam_params": best_params,
        "h_beam_verts": verts,
        "cstype": str(catalogue_df.iloc[int(best_idx)]["name"]),
        "offset": np.array([cs_cog_x, cs_cog_y], dtype=np.float64),
        "fitness": {
            "log_distance": float(best_obj[0]),
            "active_edge_length_relative": float(best_obj[1]),
            "cosine_quality": float(best_obj[2]),
            "score": float(best_score),
        },
    }
