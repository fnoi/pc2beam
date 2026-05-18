# HELIOS Data Generation

This directory is the dedicated home for controlled synthetic data generation
from IFC models using HELIOS++.

## Purpose

Use this workflow when you need reproducible simulation runs for experimentation:

1. tessellate IFC geometry and export HELIOS scene assets
2. write HELIOS scene/survey XML
3. generate per-beam IFC ground truth YAML in run output (`pc2beam_input/<ifc_stem>_gt.yaml`, with `start`/`end` XYZ endpoints and `beam_type`)
4. write beam source / reconstruction stub table YAML in `pc2beam_input/` (see Output artifact contract)
5. optionally run HELIOS to produce LAS point clouds
6. convert LAS output to pc2beam demo-compatible TXT (`xyz nx ny nz instance_id`)

Core orchestration lives in `pc2beam.helios_pipeline.run_ifc_helios_pipeline(...)`.

## Quick start

From repository root:

```bash
python tools/helios_generation/run_helios_simulation.py
```

Or use `tools/helios_generation/notebooks/simulation_demo.ipynb`.

## Canonical config and docs paths

- Scanner config:
  `tools/helios_generation/config/scanners_example.yaml`
- Simulation notebook:
  `tools/helios_generation/notebooks/simulation_demo.ipynb`

Legacy compatibility paths are currently still available at:

- `run_helios_simulation.py`
- `config/scanners_example.yaml`
- `notebooks/simulation_demo.ipynb`

## Controlled configuration knobs

Key parameters in `run_ifc_helios_pipeline(...)`:

- `include_all_geometries`
- `helios_one_obj_per_instance`
- `mesher_linear_deflection`
- `mesher_angular_deflection_deg`
- `run_simulation`
- `helios_data_path`
- `helios_extra_args`
- `run_id`
- `output_root`
- `scanners_yaml`

## Output artifact contract

Each run writes to:

`output/helios_pc2beam/<run_id>/`

Typical layout:

```text
<run_id>/
  data/
    sceneparts/pc2beam/
      instances/*.obj
      scene.obj
      scene.mtl
      scene_sidecar.yaml
    scenes/
      pc2beam_scene.xml
      pc2beam_scene.scene
  surveys/
    pc2beam_survey.xml
  sim_output/
    .../leg000_points.las
    .../leg001_points.las
  pc2beam_input/
    model_0_z_up_gt.yaml
    points_with_normals_instances.txt
    <ifc_stem>_beam_source_instance_rows.yaml
    <ifc_stem>_beam_reconstruction_io_stub.yaml
```

Beam ID tables (for joining input IFC to TXT ``instance_id`` and later to output IFC):

- **`<ifc_stem>_beam_source_instance_rows.yaml`:** ``schema_version`` + ``rows``, each row
  ``source_global_id`` (IFC ``GlobalId``), ``instance_id`` (label in
  ``points_with_normals_instances.txt``), ``bone_id``. At generation time
  ``bone_id == instance_id``; after skeleton merge, downstream tools may rewrite
  so several rows share one ``bone_id``.
- **`<ifc_stem>_beam_reconstruction_io_stub.yaml`:** same ``rows`` plus
  ``output_global_id: null`` per row for reconstruction / IFC export to fill
  (same GUID on rows that merged into one output beam).

The generated `pc2beam_input/points_with_normals_instances.txt` can be loaded with:

```python
from pc2beam.data import PointCloud

pc = PointCloud.from_txt("output/helios_pc2beam/<run_id>/pc2beam_input/points_with_normals_instances.txt")
```

## hitObjectId mapping

For `helios_one_obj_per_instance=True`, LAS `hitObjectId` is typically the
0-based scene part index. Use sidecar fields:

- `meshed_instance_ids_in_part_order`
- `instance_hit_mapping`
- `beam_instance_hit_mapping`

and helper:

- `pc2beam.ifc_mesh_export.instance_id_for_las_hit_object_id(...)`

In merged OBJ mode (`helios_one_obj_per_instance=False`), `hitObjectId` may be
constant and less informative for per-instance analysis.

## Reproducibility checklist

- Use pinned environment in `environment.yml`
- Version scanner YAML files
- Use explicit `run_id` conventions
- Record pipeline parameters, IFC source, scanner config, and git SHA
