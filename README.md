# PC2BEAM

A Python package for converting point cloud data to beam models. This package implements the methodology described in the paper ["Automated Steel Structure Model Reconstruction through Point Cloud Instance Segmentation and Parametric Shape Fitting"](https://itcon.org/papers/2025_45-ITcon-Noichl.pdf) (Journal of Information Technology in Construction, 2025).

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

You can run the interactive demo in a Google Colab notebook: https://colab.research.google.com/github/fnoi/pc2beam/blob/dev/notebooks/demo.ipynb

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/pc2beam.git
cd pc2beam
```

2. Create a virtual environment (recommended):

For Linux and MacOS:
```bash
python -m venv venv_pc2beam
source venv_pc2beam/bin/activate
```

For Windows:
```cmd
python -m venv venv_pc2beam
venv_pc2beam\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

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
  - 
- Interactive tutorial in `./notebooks/`:
  - `demo.ipynb`: Step-by-step demonstration of the package functionality

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