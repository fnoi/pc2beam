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

2. Create the Conda environment (Python 3.12):
```bash
conda env create -f environment.yml
```

3. Activate the environment:
```bash
conda activate pc2beam
```

4. Verify the installation:
```bash
python -c "import open3d, numpy, plotly, sklearn, omegaconf, matplotlib, tqdm, ifcopenshell, laspy; print('pc2beam env ready')"
```

### Reproducible installs (Conda vs pip)

- **Full environment (IFC + HELIOS++):** use [environment.yml](environment.yml). `helios` and `ifcopenshell` are **pinned** so `conda env create -f environment.yml` stays on the same major/minor releases. Build hashes still differ by platform (e.g. Linux vs macOS); the file was validated with `conda env create --dry-run` on **macOS arm64**.
- **Light install (no HELIOS binary):** [requirements.txt](requirements.txt) plus `pip install -e .` — suitable for Google Colab and machines where you only need Python libraries. There is no pip-distributed `helios` equivalent; keep simulation off (`RUN_HELIOS = False` in `simulation_demo.ipynb`).

Notes:
- `environment.yml` already installs `pc2beam` in editable mode via `pip -e .`; no extra install step is required.
- If dependency resolution fails for `open3d` on your platform with Python 3.12, use Python 3.11 instead:
```bash
conda create -n pc2beam python=3.11 numpy open3d plotly scikit-learn omegaconf matplotlib tqdm pip -c conda-forge
conda activate pc2beam
pip install -e .
```

### Troubleshooting

- **`ModuleNotFoundError: No module named 'laspy'`** (or other missing deps): the notebook or terminal is using a different Python than your Conda env. In a notebook run `import sys; print(sys.executable)` and confirm it matches `which python` after `conda activate pc2beam`. Fix: `conda install -n pc2beam -c conda-forge laspy` or `conda env update -f environment.yml --prune`; for pip-only setups use `pip install laspy` or `pip install -r requirements.txt`. In VS Code / Cursor / Jupyter, **select the `pc2beam` kernel** (Python 3.12 from that env). If the env does not appear, register it once: `conda activate pc2beam && python -m ipykernel install --user --name pc2beam --display-name "Python (pc2beam)"`.

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
  - `simulation_demo.ipynb`: IFC tessellation, HELIOS++ survey generation, and optional simulated TLS output

## IFC models and HELIOS++ laser scanning

The repository includes `data/model_0_z_up.ifc` (IFC4 steel frame). The package can tessellate **IfcBeam** geometry to Wavefront OBJ/MTL (one material per profile/material combination), write HELIOS++ scene and survey XML, and drive the external **`helios`** simulator.

1. Use the Conda environment above — **`helios` is already pinned** in `environment.yml`. To add HELIOS to another env: `conda install -c conda-forge helios=2.1.0` (match the pin when possible).
2. Stock HELIOS assets (platforms, scanners, …): conda-forge often ships them under **`$CONDA_PREFIX/share/helios`** or inside the **`pyhelios`** package (`…/site-packages/pyhelios/`). `pc2beam` resolves this automatically when possible. Set **`HELIOS_DATA_PATH`** only if you use a custom HELIOS data checkout (directory whose `data/` tree contains `platforms.xml`).
3. Check the installation:
   ```bash
   helios --test
   ```
4. Edit `config/scanners_example.yaml` for scanner standpoints and sweep parameters, then run:
   ```bash
   python run_helios_simulation.py
   ```
   or follow `notebooks/simulation_demo.ipynb`. Outputs are written under `output/helios_pc2beam/<run_id>/` (ignored by git when under `output/`).

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