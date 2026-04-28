"""Tests for adaptive S2 threshold suitability logic."""

import unittest

import numpy as np

from pc2beam.processing import calculate_s2


def _make_l_beam_points(
    n_per_face: int = 120,
    length: float = 2.0,
    noise_sigma: float = 0.002,
    seed: int = 42,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    z = rng.uniform(-length / 2.0, length / 2.0, n_per_face)
    # Two orthogonal faces: x=0 plane and y=0 plane.
    x_face = np.column_stack(
        [
            rng.normal(0.0, noise_sigma, n_per_face),
            rng.uniform(-0.08, 0.08, n_per_face),
            z,
        ]
    )
    y_face = np.column_stack(
        [
            rng.uniform(-0.08, 0.08, n_per_face),
            rng.normal(0.0, noise_sigma, n_per_face),
            z,
        ]
    )
    return np.vstack([x_face, y_face]).astype(np.float64)


class TestS2ThresholdQuality(unittest.TestCase):
    def test_backward_compatible_single_threshold(self):
        points = _make_l_beam_points()
        instances = np.zeros(len(points), dtype=np.int32)
        out = calculate_s2(
            points=points,
            instances=instances,
            distance_threshold=0.02,
            ransac_n=3,
            num_iterations=800,
            min_plane_inliers=20,
            enable_fallback=False,
        )
        feature = out[0]
        self.assertEqual(feature["status"], "ok")
        self.assertAlmostEqual(feature["selected_distance_threshold"], 0.02, places=6)
        self.assertEqual(feature["threshold_attempts"], 1)
        self.assertIsNotNone(feature["projection_residual_median"])
        self.assertIsNotNone(feature["projection_support_ratio"])

    def test_strict_first_schedule_selection(self):
        points = _make_l_beam_points(noise_sigma=0.0015)
        instances = np.zeros(len(points), dtype=np.int32)
        out = calculate_s2(
            points=points,
            instances=instances,
            distance_threshold=0.02,
            distance_threshold_schedule=[0.004, 0.008, 0.02],
            ransac_n=3,
            num_iterations=1000,
            min_plane_inliers=20,
            enable_fallback=False,
        )
        feature = out[0]
        self.assertEqual(feature["status"], "ok")
        self.assertAlmostEqual(feature["selected_distance_threshold"], 0.004, places=6)
        self.assertEqual(feature["threshold_attempts"], 1)
        self.assertGreaterEqual(feature["projection_support_ratio"], 0.35)
        self.assertGreaterEqual(feature["suitability_score"], 0.0)
        self.assertLessEqual(feature["suitability_score"], 1.0)

    def test_fallback_when_all_quality_gates_fail(self):
        points = _make_l_beam_points(noise_sigma=0.0025)
        instances = np.zeros(len(points), dtype=np.int32)
        out = calculate_s2(
            points=points,
            instances=instances,
            distance_threshold=0.02,
            distance_threshold_schedule=[0.004, 0.008],
            ransac_n=3,
            num_iterations=800,
            min_plane_inliers=20,
            quality={"support_ratio_min": 0.9999},
            enable_fallback=True,
        )
        feature = out[0]
        self.assertEqual(feature["status"], "fallback_pca")
        self.assertIsNone(feature["selected_distance_threshold"])
        self.assertEqual(feature["threshold_attempts"], 2)
        self.assertIsNone(feature["projection_residual_p90"])


if __name__ == "__main__":
    unittest.main()
