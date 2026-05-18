import unittest

from omegaconf import OmegaConf

from pc2beam.helios_pipeline import (
    _build_beam_reconstruction_io_stub_rows,
    _build_beam_source_instance_rows,
)


class TestBeamSourceInstanceRows(unittest.TestCase):
    def test_build_rows_skips_empty_global_id(self):
        sidecar = OmegaConf.create(
            {
                "beams": [
                    {"global_id": "G1", "instance_id": 1},
                    {"global_id": "", "instance_id": 2},
                    {"global_id": None, "instance_id": 3},
                    {"global_id": "None", "instance_id": 4},
                    {"global_id": "G2", "instance_id": 5},
                ]
            }
        )
        rows = _build_beam_source_instance_rows(sidecar)
        self.assertEqual(
            rows,
            [
                {"source_global_id": "G1", "instance_id": 1, "bone_id": 1},
                {"source_global_id": "G2", "instance_id": 5, "bone_id": 5},
            ],
        )

    def test_build_rows_coerces_instance_id(self):
        sidecar = OmegaConf.create({"beams": [{"global_id": "G1", "instance_id": "7"}]})
        rows = _build_beam_source_instance_rows(sidecar)
        self.assertEqual(
            rows,
            [{"source_global_id": "G1", "instance_id": 7, "bone_id": 7}],
        )

    def test_stub_rows_add_output_global_id_none(self):
        rows = [
            {"source_global_id": "G1", "instance_id": 1, "bone_id": 1},
            {"source_global_id": "G2", "instance_id": 2, "bone_id": 2},
        ]
        stub = _build_beam_reconstruction_io_stub_rows(rows)
        self.assertEqual(
            stub,
            [
                {"source_global_id": "G1", "instance_id": 1, "bone_id": 1, "output_global_id": None},
                {"source_global_id": "G2", "instance_id": 2, "bone_id": 2, "output_global_id": None},
            ],
        )


if __name__ == "__main__":
    unittest.main()
