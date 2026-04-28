# `demo_script.py` Quick Guide

This file documents how to run and configure `demo_script.py`, a script-first demo entry point around `run_pc2beam(...)`.

## What It Does

`demo_script.py` runs an end-to-end pipeline on a point cloud and can:

- compute `s2` centerline estimation
- run legacy projection (`features["legacy_projection"]`)
- run catalogue-based cross-section fitting
- export Plotly HTML visualizations
- generate a 3x3 cross-section grid artifact (`cross_sections_9.html`)
- optionally evaluate against GT YAML and write per-instance metrics CSV

It is intended as a stable alternative to notebook execution.

## Inputs

- Point cloud TXT (`x y z [nx ny nz] [instance_id]`)
- Config YAML (default: `config/default.yaml`)
- Optional GT YAML for `--evaluate`
- Optional catalogue CSV override for catalogue fitting

If `--points` is not provided, the script tries:
1. latest HELIOS output: `output/helios_pc2beam/<run>/pc2beam_input/points_with_normals_instances.txt`
2. fallback: `data/test_points.txt`

## Outputs

When visualization is enabled (default inside the script), outputs are written under:

- `output/demo_runs/<UTC timestamp>/` (or `--viz-dir`)

Typical artifacts include:

- `*_point_cloud_by_instance.html`
- `*_skeleton.html`
- `*_skeleton_with_points.html`
- `cross_sections_9.html` (with projection-line overlays when projection metadata exists)

Optional:

- per-instance evaluation CSV via `--metrics-csv` (requires `--evaluate`)

## Current Script Defaults

In `main()`, this demo currently forces:

- `args.viz = True`
- `args.cross_sections = True`
- `args.legacy = True`
- `args.catalogue = True`

So running `python demo_script.py` is already a full demo flow.

## Common Commands

Run with defaults:

```bash
python demo_script.py
```

Use explicit point cloud and config:

```bash
python demo_script.py --points data/test_points.txt --config config/default.yaml
```

Enable evaluation and write metrics:

```bash
python demo_script.py --evaluate --gt-yaml data/test_points_gt.yaml --metrics-csv output/demo_metrics.csv
```

Override catalogue CSV path:

```bash
python demo_script.py --catalogue-csv-path data/European_Steel_Section_Properties.csv
```

Tune catalogue fitting overrides from CLI:

```bash
python demo_script.py --catalogue-n-downsample 200 --polygon-subdivision
```

## Key Configuration Knobs

From `config/default.yaml`:

- `s2.*`: RANSAC and threshold-suitability settings
- `legacy_projection.*`: projection behavior used for 2D section data
- `catalogue_fit.*`: catalogue source and optimization settings

CLI overrides in `demo_script.py` are merged with config values for selected options.

## Notes

- The script expects runtime dependencies (`open3d`, `scikit-learn`, plotting stack) to be available in the active environment.
- For reproducible artifacts, keep config and input file paths explicit in command logs.
