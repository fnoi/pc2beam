"""Tests for beam cross-section (legacy projection) pipeline."""

import unittest
from pathlib import Path

import numpy as np

from pc2beam.data import PointCloud
from pc2beam.processing import project_instance_to_section_2d


class TestBeamSectionProjection(unittest.TestCase):
    def test_synthetic_elongated_cloud_ok_shapes(self):
        rng = np.random.default_rng(42)
        n = 120
        pts = np.column_stack(
            [
                rng.normal(0, 0.02, n),
                rng.normal(0, 0.02, n),
                rng.uniform(-0.6, 0.6, n),
            ]
        ).astype(np.float64)
        nrm = np.tile(np.array([0.0, 1.0, 0.0], dtype=np.float64), (n, 1))
        out = project_instance_to_section_2d(
            pts,
            nrm,
            distance_threshold=0.02,
            min_plane_inliers=8,
        )
        self.assertTrue(out.get("ok"), msg=str(out))
        self.assertEqual(out["points_2d"].shape, (n, 2))
        self.assertEqual(out["normals_2d"].shape, (n, 2))
        self.assertTrue(np.isfinite(out["points_2d"]).all())
        self.assertTrue(np.isfinite(out["normals_2d"]).all())

    def test_insufficient_points(self):
        pts = np.zeros((1, 3), dtype=np.float64)
        nrm = np.zeros((1, 3), dtype=np.float64)
        out = project_instance_to_section_2d(pts, nrm)
        self.assertFalse(out["ok"])
        self.assertEqual(out["status"], "insufficient_points")

    def test_data_test_points_legacy_projection(self):
        path = Path(__file__).resolve().parents[1] / "data" / "test_points.txt"
        pc = PointCloud.from_txt(path)
        pc.compute_s2()
        pc.compute_legacy_projection()
        self.assertIn("legacy_projection", pc.features)
        ok_count = sum(1 for v in pc.features["legacy_projection"].values() if v.get("ok"))
        self.assertGreater(ok_count, 0)


if __name__ == "__main__":
    unittest.main()
