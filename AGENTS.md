# AGENTS.md

## Project Description

`pc2beam` is a Python 3.11 research codebase for reconstructing beam-like structural models from segmented point clouds.  
It also includes an IFC-to-HELIOS++ synthetic data generation workflow that can produce demo-compatible point cloud input files and IFC-derived beam ground truth.

Primary domains in this repository:

- Core point-cloud processing and skeleton reconstruction (`pc2beam/`)
- Evaluation against YAML ground truth (`pc2beam/evaluation.py`)
- IFC mesh export, HELIOS++ scene/survey generation, and LAS-to-pc2beam conversion (`pc2beam/helios_*`, `pc2beam/ifc_*`)
- Interactive notebooks for demonstrations (`notebooks/`, `tools/helios_generation/notebooks/`)

## Repository Layout

- `pc2beam/`: core package modules
- `run.py`: main functional entry point for point-cloud pipeline (`run_pc2beam(...)`)
- `config/default.yaml`: default processing configuration for s1/s2 and optional flows
- `tools/helios_generation/`: canonical HELIOS workflow docs, scripts, and scanner config
- `data/`: sample inputs and sample ground truth YAML files
- `output/`: generated artifacts (large; treat as derived data)
- `notebooks/`: user-facing demos (including compatibility copies)

## Core Execution Paths

### 1) Point-cloud pipeline (primary package flow)

Use `run.py` / `run_pc2beam(...)` for:

- loading a TXT point cloud (`x y z [nx ny nz] [instance_id]`)
- computing s1 and s2 features
- optional legacy projection
- optional catalogue-based cross-section fitting
- optional evaluation against ground truth YAML
- skeleton visualization

### 2) IFC + HELIOS synthetic generation flow

Use `pc2beam.helios_pipeline.run_ifc_helios_pipeline(...)` (or wrapper script in `tools/helios_generation/`) for:

- IFC tessellation and OBJ/MTL export
- scene/survey XML generation
- IFC beam GT YAML export into `pc2beam_input/<ifc_stem>_gt.yaml`
- optional HELIOS simulation
- LAS conversion to `points_with_normals_instances.txt`

## Environment and Dependencies

Preferred environment:

- `conda env create -f environment.yml`
- `conda activate pc2beam`

Notes:

- `environment.yml` is the canonical full environment (includes pinned `helios` and `ifcopenshell`).
- `requirements.txt` is suitable for lighter/pip-oriented installs where HELIOS simulation may be disabled.
- The repository targets Python 3.11.

## Implementation Instructions

These instructions are for coding agents and contributors making changes in this repository.

1. Keep changes minimal and scoped to the requested behavior.
2. Preserve existing APIs unless the task explicitly requests a breaking change.
3. Prefer extending current modules over creating parallel duplicate flows.
4. Keep paths and artifact contracts stable for notebooks and downstream scripts.
5. Treat files under `output/` as generated artifacts; avoid editing them manually.

### Code Style and Design Expectations

- Follow existing style in each file (type hints are present in many newer modules; keep them where applicable).
- Prefer small, testable functions for new logic.
- Raise explicit `ValueError` with clear messages for invalid user input/configuration.
- Keep pure computation separate from visualization and IO when practical.
- Avoid silent behavior changes in geometry/projection/fit pipelines.

### Data and File Contract Expectations

- Maintain TXT point-cloud column contract:
  - required: `x y z`
  - optional normals: `nx ny nz`
  - optional instance label: `instance_id`
- Maintain HELIOS output contract under `output/helios_pc2beam/<run_id>/...`.
- Preserve ground-truth YAML schema (`start`, `end`, `beam_type`) unless migration is explicitly requested.

### Configuration Changes

When editing config-driven behavior:

- update defaults in `config/default.yaml` only when justified
- keep backward-compatible config keys
- prefer reading values with safe fallbacks
- document new knobs in relevant README/docs

### Notebook and Script Compatibility

- `tools/helios_generation/` is the canonical location for HELIOS workflow assets.
- Compatibility copies exist at:
  - `run_helios_simulation.py`
  - `notebooks/simulation_demo.ipynb`
  - `config/scanners_example.yaml`
- If canonical behavior changes, update compatibility entry points or document intentional divergence.

## Validation Checklist for Changes

For most code changes, run the smallest relevant checks:

- Import smoke test:
  - `python -c "import pc2beam; print('ok')"`
- Main pipeline smoke run (sample data):
  - `python run.py`
- If HELIOS-related code changed, run at least setup/no-sim path using the HELIOS generation script and verify output paths.
- If evaluation code changed, verify against sample ground truth YAML in `data/`.

## Common Pitfalls

- Assuming normals/instances always exist: guard with validation where needed.
- Breaking notebook expectations by changing output paths or schemas without updates.
- Mixing canonical and compatibility HELIOS paths inconsistently.
- Introducing heavyweight dependencies without updating environment files.

## Guidance for Future Contributors

- Prefer documenting behavior changes in `README.md` and `tools/helios_generation/README.md`.
- Keep research reproducibility in mind: deterministic settings, explicit config, and clear output contracts.
- When uncertain, preserve existing artifact shapes and add optional fields instead of replacing current ones.
