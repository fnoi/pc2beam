"""Tests for scanner-aware normal orientation in HELIOS LAS conversion."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from omegaconf import OmegaConf

from pc2beam.helios_to_pc2beam import (
    _orient_normals_towards_scanner,
    export_helios_sim_to_pc2beam_txt,
)


class TestHeliosNormalsOrientation(unittest.TestCase):
    def test_orient_normals_towards_scanner_flips_away_normals(self):
        points = np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float32)
        normals = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]], dtype=np.float32)
        scanner_position = np.array([0.0, 0.0, 0.0], dtype=np.float64)

        oriented = _orient_normals_towards_scanner(points, normals, scanner_position)
        vectors_to_scanner = scanner_position[None, :] - points
        dot = np.sum(oriented * vectors_to_scanner, axis=1)

        self.assertTrue(np.all(dot >= 0.0))

    def test_export_uses_scanner_oriented_per_leg_normals(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            las0 = tmp / "leg000_points.las"
            las1 = tmp / "leg001_points.las"
            las0.write_bytes(b"")
            las1.write_bytes(b"")

            sidecar = OmegaConf.create({"beam_instance_hit_mapping": []})
            output_txt = tmp / "points_with_normals_instances.txt"

            leg0_points = np.array(
                [[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.0, 0.2, 0.0], [0.2, 0.2, 0.0]],
                dtype=np.float32,
            )
            leg1_points = np.array(
                [[10.0, 0.0, 0.0], [10.2, 0.0, 0.0], [10.0, 0.2, 0.0], [10.2, 0.2, 0.0]],
                dtype=np.float32,
            )

            scan_by_name = {
                las0.name: SimpleNamespace(points=leg0_points, hit_object_id=None),
                las1.name: SimpleNamespace(points=leg1_points, hit_object_id=None),
            }

            def _fake_read_las_scan(path):
                return scan_by_name[Path(path).name]

            summary = None
            with patch("pc2beam.helios_to_pc2beam.read_las_scan", side_effect=_fake_read_las_scan):
                summary = export_helios_sim_to_pc2beam_txt(
                    sim_output_dir=tmp,
                    sidecar=sidecar,
                    output_txt_path=output_txt,
                    scanner_positions=[[0.0, 0.0, 1.0], [11.0, 0.0, 1.0]],
                    orient_towards_scanner=True,
                    per_leg_normals=True,
                )

            data = np.loadtxt(output_txt)
            normals = data[:, 3:6]
            points = data[:, 0:3]

            # First N points belong to leg0, remaining to leg1 due to sorted leg files.
            n0 = len(leg0_points)
            dots0 = np.sum(normals[:n0] * (np.array([0.0, 0.0, 1.0]) - points[:n0]), axis=1)
            dots1 = np.sum(normals[n0:] * (np.array([11.0, 0.0, 1.0]) - points[n0:]), axis=1)
            self.assertTrue(np.all(dots0 >= -1e-6))
            self.assertTrue(np.all(dots1 >= -1e-6))

            self.assertTrue(summary["per_leg_normals"])
            self.assertTrue(summary["scanner_oriented_normals_used"])
            self.assertIsNone(summary["scanner_orientation_fallback_reason"])

    def test_export_falls_back_without_scanner_positions(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            las0 = tmp / "leg000_points.las"
            las0.write_bytes(b"")

            sidecar = OmegaConf.create({"beam_instance_hit_mapping": []})
            output_txt = tmp / "points_with_normals_instances.txt"
            points = np.array(
                [[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.0, 0.2, 0.0], [0.2, 0.2, 0.0]],
                dtype=np.float32,
            )

            with patch(
                "pc2beam.helios_to_pc2beam.read_las_scan",
                return_value=SimpleNamespace(points=points, hit_object_id=None),
            ):
                summary = export_helios_sim_to_pc2beam_txt(
                    sim_output_dir=tmp,
                    sidecar=sidecar,
                    output_txt_path=output_txt,
                    scanner_positions=None,
                    orient_towards_scanner=True,
                    per_leg_normals=True,
                )

            self.assertFalse(summary["scanner_oriented_normals_used"])
            self.assertEqual(
                summary["scanner_orientation_fallback_reason"],
                "scanner_positions_not_provided",
            )
            self.assertEqual(summary["leg_count"], 1)
            self.assertEqual(summary["scanner_position_count"], 0)


if __name__ == "__main__":
    unittest.main()
