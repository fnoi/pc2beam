"""Tests for CSV catalogue loading and normalization."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from pc2beam.ifc_io import load_ishape_catalogue, resolve_catalogue_csv_path


class TestCatalogueLoader(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.eur_csv = self.repo_root / "data" / "European_Steel_Section_Properties.csv"
        self.us_csv = self.repo_root / "data" / "aisc-shapes-database-v15.0.csv"

    def test_resolve_region_defaults(self):
        eur = resolve_catalogue_csv_path(catalogue_region="eur")
        us = resolve_catalogue_csv_path(catalogue_region="us")
        self.assertEqual(eur, self.eur_csv)
        self.assertEqual(us, self.us_csv)

    def test_load_eur_catalogue(self):
        _, df = load_ishape_catalogue(self.eur_csv, catalogue_region="eur")
        self.assertFalse(df.empty)
        self.assertTrue({"name", "tw", "tf", "bf", "d"}.issubset(set(df.columns)))
        self.assertTrue((df[["tw", "tf", "bf", "d"]] > 0.0).all().all())

    def test_load_us_catalogue(self):
        _, df = load_ishape_catalogue(self.us_csv, catalogue_region="us")
        self.assertFalse(df.empty)
        self.assertTrue({"name", "tw", "tf", "bf", "d"}.issubset(set(df.columns)))
        self.assertTrue((df[["tw", "tf", "bf", "d"]] > 0.0).all().all())

    def test_invalid_region_raises(self):
        with self.assertRaises(ValueError):
            resolve_catalogue_csv_path(catalogue_region="apac")

    def test_missing_columns_raises(self):
        with tempfile.TemporaryDirectory() as td:
            bad = Path(td) / "bad.csv"
            pd.DataFrame({"foo": [1], "bar": [2]}).to_csv(bad, index=False)
            with self.assertRaises(ValueError):
                load_ishape_catalogue(bad, catalogue_region="eur")


if __name__ == "__main__":
    unittest.main()
