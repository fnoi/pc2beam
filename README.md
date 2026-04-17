# PC2BEAM

A Python package for converting point cloud data to beam models. This package implements the methodology described in the paper ["Automated Steel Structure Model Reconstruction through Point Cloud Instance Segmentation and Parametric Shape Fitting"](https://itcon.org/papers/2025_45-ITcon-Noichl.pdf).

If you use this code in your academic work, please cite the paper using the citation information provided at the bottom of this README.

## Overview

This package provides tools and utilities for processing point cloud data and converting it into beam models. The methodology includes:

- Point cloud processing
  - Local neighborhood orientation estimation using supernormal $\vec{s_1}$
  - Segment-level orientation estimation $\vec{s_2}$ and point projection
  - Cross-section fitting using multi-objective optimization from standardized catalog
- Model reconstruction
- Tools for evaluation and interactive visualization intermediate and final results

## Demo

You can run the interactive demo in a Google Colab notebook: https://colab.research.google.com/github/fnoi/pc2beam/blob/main/notebooks/demo.ipynb

## Installation

1. Clone the repository:
```bash
git clone https://github.com/fnoi/pc2beam.git
cd pc2beam
```

2. Create the Conda environment (Python 3.11):
```bash
conda env create -f environment.yml
```

3. Activate the environment:
```bash
conda activate pc2beam
```

4. Verify the installation:
```bash
python -c "import pyhelios, run_helios_simulation, open3d, numpy, plotly, sklearn, omegaconf, matplotlib, tqdm, ifcopenshell, laspy; print('pc2beam env ready')"
```

### Reproducible installs (Conda vs pip)

- **Full environment (IFC + HELIOS++):** use [environment.yml](environment.yml). `helios` and `ifcopenshell` are **pinned** so `conda env create -f environment.yml` stays on the same major/minor releases. Build hashes still differ by platform (e.g. Linux vs macOS); the file is validated against the `pc2beam` environment on **macOS arm64**.
- **Light install (no HELIOS binary):** [requirements.txt](requirements.txt) plus `pip install -e .` — suitable for Google Colab and machines where you only need Python libraries. There is no pip-distributed `helios` equivalent; keep simulation off (`RUN_HELIOS = False` in `simulation_demo.ipynb`).

Notes:
- `environment.yml` already installs `pc2beam` in editable mode via `pip -e .`; no extra install step is required.
- `pc2beam` is the canonical project environment for this repository.

### Troubleshooting

- **`ModuleNotFoundError: No module named 'laspy'`** (or other missing deps): the notebook or terminal is using a different Python than your Conda env. In a notebook run `import sys; print(sys.executable)` and confirm it matches `which python` after `conda activate pc2beam`. Fix: `conda install -n pc2beam -c conda-forge laspy` or `conda env update -n pc2beam -f environment.yml --prune`; for pip-only setups use `pip install laspy` or `pip install -r requirements.txt`. In VS Code / Cursor / Jupyter, **select the `pc2beam` kernel** (Python 3.11 from that env). If the env does not appear, register it once: `conda activate pc2beam && python -m ipykernel install --user --name pc2beam --display-name "Python (pc2beam)"`.

## Usage

The package processes two types of input files:

1. Point cloud data (*.txt format) containing:
   - Coordinates $(X, Y, Z)$
   - Normal vectors $(N_x, N_y, N_z)$
   - Instance labels $l_i$

2. Steel profile catalog (*.csv format) with standardized cross-section definitions

Example files are provided in the repository:
- Data files in `./data/`:
  - `test_points.txt`: Sample point cloud data
  - `profiles.csv`: Sample steel profile catalog
  - `model_0_z_up.ifc`: Sample IFC4 structural model for the HELIOS++ workflow
- Interactive tutorial in `./notebooks/`:
  - `demo.ipynb`: Step-by-step demonstration of the package functionality
  - `simulation_demo.ipynb`: compatibility copy of the HELIOS simulation notebook
- HELIOS generation docs and canonical assets in `./tools/helios_generation/`:
  - `README.md`: dedicated workflow documentation
  - `notebooks/simulation_demo.ipynb`: canonical HELIOS simulation notebook

## IFC models and HELIOS++ laser scanning

The HELIOS synthetic data-generation workflow has a dedicated home in:

- [`tools/helios_generation/README.md`](tools/helios_generation/README.md)

Use those docs for setup, controlled generation knobs, artifact semantics,
`hitObjectId` mapping, and reproducibility guidance.

Primary entry assets:

- Script: [`tools/helios_generation/run_helios_simulation.py`](tools/helios_generation/run_helios_simulation.py)
- Notebook: [`tools/helios_generation/notebooks/simulation_demo.ipynb`](tools/helios_generation/notebooks/simulation_demo.ipynb)
- Scanner config: [`tools/helios_generation/config/scanners_example.yaml`](tools/helios_generation/config/scanners_example.yaml)

When the IFC pipeline runs, it also auto-generates per-beam ground truth into
the run output as
`output/helios_pc2beam/<run_id>/pc2beam_input/<ifc_stem>_gt.yaml`.
For the sample IFC this is `model_0_z_up_gt.yaml`. The file contains per-beam
line endpoints as `start: [x, y, z]`, `end: [x, y, z]`, plus `beam_type`.

When HELIOS simulation is enabled, the pipeline additionally writes a
demo-compatible input TXT at
`output/helios_pc2beam/<run_id>/pc2beam_input/points_with_normals_instances.txt`
with columns: `x y z nx ny nz instance_id`. This can be used directly with
`PointCloud.from_txt(...)` as in `notebooks/demo.ipynb`.

Compatibility paths remain available:

- `run_helios_simulation.py`
- `notebooks/simulation_demo.ipynb`
- `config/scanners_example.yaml`

## License

This project is licensed under the [MIT License](LICENSE).

## BibTex

```bibtex
@article{2025_PC2BEAM,
   author = {Florian Noichl and Yuandong Pan and André Borrmann},
   doi = {10.36680/j.itcon.2025.045},
   issn = {1874-4753},
   journal = {Journal of Information Technology in Construction},
   month = {7},
   pages = {1099-1122},
   title = {Automated Steel Structure Model Reconstruction through Point Cloud Instance Segmentation and Parametric Shape Fitting},
   volume = {30},
   url = {https://itcon.org/paper/2025/45},
   year = {2025}
}
```