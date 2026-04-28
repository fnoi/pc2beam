"""Tests for projection-line overlays in 2D cross-section grid."""

import unittest
from types import SimpleNamespace

import numpy as np

from pc2beam import viz


class TestProjectionLineOverlay(unittest.TestCase):
    def _make_point_cloud(self, projection_entry):
        return SimpleNamespace(
            features={
                "legacy_projection": {1: projection_entry},
                "catalogue_fit": {},
            }
        )

    def test_plot_cross_section_grid_with_projection_lines(self):
        points_2d = np.array(
            [[-0.1, -0.05], [0.05, 0.1], [0.1, -0.02], [-0.08, 0.08]],
            dtype=np.float64,
        )
        projection = {
            "ok": True,
            "points_2d": points_2d,
            "transform": np.eye(4, dtype=np.float64),
            "proj_dir_0": np.array([1.0, 0.0, 0.0], dtype=np.float64),
            "proj_dir_1": np.array([0.0, 1.0, 0.0], dtype=np.float64),
        }
        pc = self._make_point_cloud(projection)

        fig = viz.plot_cross_section_grid(
            pc,
            [1],
            show_projection_lines=True,
        )
        # 1 points trace + 2 line traces
        self.assertGreaterEqual(len(fig.data), 3)
        line_traces = [tr for tr in fig.data if getattr(tr, "mode", None) == "lines"]
        self.assertGreaterEqual(len(line_traces), 2)

    def test_plot_cross_section_grid_skips_missing_projection_data(self):
        points_2d = np.array([[0.0, 0.0], [0.1, 0.1], [-0.05, 0.02]], dtype=np.float64)
        projection = {
            "ok": True,
            "points_2d": points_2d,
            # Missing transform/proj_dir_* on purpose; should gracefully skip lines.
        }
        pc = self._make_point_cloud(projection)

        fig = viz.plot_cross_section_grid(
            pc,
            [1],
            show_projection_lines=True,
        )
        # Only points trace should be present when no line metadata is available.
        self.assertEqual(len(fig.data), 1)
        self.assertEqual(getattr(fig.data[0], "mode", None), "markers")


if __name__ == "__main__":
    unittest.main()
