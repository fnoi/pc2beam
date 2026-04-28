import numpy as np
import pytest

from pc2beam.catalog_fit import FitConfig, _cost_combined, params2verts, solve_w_nsga_style
from pc2beam.cs_geometry import kmeans_points_normals_2D, subdivide_edges


def test_subdivide_edges_short_unchanged():
    edges = np.array([[[0.0, 0.0], [0.005, 0.0]]], dtype=np.float64)
    n = np.array([[0.0, 1.0]], dtype=np.float64)
    out_e, out_n = subdivide_edges(edges, n, lmax=0.01)
    assert out_e.shape == edges.shape
    np.testing.assert_array_almost_equal(out_e, edges)


def test_subdivide_edges_splits_long_edge():
    edges = np.array([[[0.0, 0.0], [0.05, 0.0]]], dtype=np.float64)
    n = np.array([[0.0, 1.0]], dtype=np.float64)
    out_e, out_n = subdivide_edges(edges, n, lmax=0.01)
    assert out_e.shape[0] > 1
    lengths = np.linalg.norm(out_e[:, 1] - out_e[:, 0], axis=1)
    assert np.all(lengths <= 0.01 + 1e-9)
    assert out_n.shape[0] == out_e.shape[0]


def test_kmeans_weights_sum_and_determinism():
    rng = np.random.default_rng(0)
    pts = rng.standard_normal((50, 2))
    nrm = rng.standard_normal((50, 2))
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
    k = 10
    rep, rn, w, labels = kmeans_points_normals_2D(pts, nrm, k, random_state=42)
    assert rep.shape == (k, 2)
    assert rn.shape == (k, 2)
    assert w.shape == (k,)
    assert abs(float(np.sum(w)) - 50.0) < 1e-6
    rep2, rn2, w2, labels2 = kmeans_points_normals_2D(pts, nrm, k, random_state=42)
    np.testing.assert_array_equal(labels, labels2)
    np.testing.assert_array_almost_equal(w, w2)


def test_kmeans_rejects_too_many_clusters():
    pts = np.zeros((5, 2))
    nrm = np.tile(np.array([[1.0, 0.0]]), (5, 1))
    with pytest.raises(ValueError):
        kmeans_points_normals_2D(pts, nrm, 10)


def test_cost_combined_weights_change_objective():
    params = np.array([0.0, 0.0, 0.02, 0.02, 0.2, 0.3], dtype=np.float64)
    pts = np.array([[0.0, 0.05], [0.2, 0.0]], dtype=np.float64)
    nrm = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64)
    a = _cost_combined(
        params,
        pts,
        nrm,
        polygon_subdivision=False,
        cluster_weights=None,
        use_cluster_weights=True,
    )
    b = _cost_combined(
        params,
        pts,
        nrm,
        polygon_subdivision=False,
        cluster_weights=np.array([2.0, 1.0]),
        use_cluster_weights=True,
    )
    assert a[0] != b[0] or a[2] != b[2]


@pytest.mark.parametrize("poly", [False, True])
def test_solve_w_nsga_style_smoke(poly):
    import pandas as pd

    df = pd.DataFrame(
        [
            {"name": "T1", "tw": 0.01, "tf": 0.01, "bf": 0.1, "d": 0.1},
            {"name": "T2", "tw": 0.02, "tf": 0.02, "bf": 0.15, "d": 0.12},
        ]
    )
    rng = np.random.default_rng(1)
    pts = rng.standard_normal((24, 2)) * 0.02
    nrm = rng.standard_normal((24, 2))
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
    cfg = FitConfig(n_pop=8, n_gen=2, polygon_subdivision=poly, edge_subdivision_lmax=0.01)
    out = solve_w_nsga_style(pts, nrm, df, fit_cfg=cfg)
    assert out["h_beam_params"].shape == (6,)
    assert out["h_beam_verts"].shape[1] == 2
