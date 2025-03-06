# PC2Beam

A Python package for converting point cloud data to beam models.

## Introduction

This package implements the methodology described in the paper ["Automated Steel Structure Model Reconstruction through Point Cloud Instance Segmentation and Parametric Shape Fitting"](https://arxiv.org/abs/2403.XXXXX) (2025, under review).

If you use this code in your academic work, please cite the paper using the citation information provided at the bottom of this README.

## Overview

This package provides tools and utilities for processing point cloud data and converting it into beam models. The methodology includes:

- Local neighborhood orientation estimation
- Segment-level orientation estimation
- Point projection and parametric shape fitting from standardized catalog
- Model reconstruction and export to IFC format

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

[Add usage examples and documentation here]

## Features

- Point cloud data processing
- Beam model generation
- [Add more features as they are implemented]

## Development

The development environment uses the same dependencies as the main package. To get started:

1. Create and activate a virtual environment as described in the Installation section
2. Install dependencies using `pip install -r requirements.txt`

## Testing

Run tests using:
```bash
pytest
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the [MIT License](LICENSE).

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{pc2beam2025,
  title={Automated Steel Structure Model Reconstruction through Point Cloud Instance Segmentation and Parametric Shape Fitting},
  author={[Authors]},
  journal={[Journal Name]},
  year={2025},
  publisher={[Publisher]},
  doi={[DOI]},
  url={https://arxiv.org/abs/2403.XXXXX}
}
```