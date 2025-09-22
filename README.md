# Supplementary Analysis Scripts for kh2d-solver Paper

[![DOI](https://zenodo.org/badge/1055379655.svg)](https://doi.org/10.5281/zenodo.17161402)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: WTFPL](https://img.shields.io/badge/License-WTFPL-brightgreen.svg)](http://www.wtfpl.net/about/)
[![Paper](https://img.shields.io/badge/Paper-arXiv:2509.16080-b31b1b.svg)](http://arxiv.org/abs/2509.16080)
[![Data Archive](https://img.shields.io/badge/Data-OSF.IO/HF6KX-blue)](https://doi.org/10.17605/OSF.IO/HF6KX)

Supplementary analysis and visualization scripts for the paper *"kh2d-solver: A Python Library for Idealized Two-Dimensional Incompressible Kelvin-Helmholtz Instability"*

## Description

This repository contains the post-processing and statistical analysis scripts used to generate figures and statistics from the kh2d-solver simulations presented in the paper. These scripts process NetCDF output files from the main solver to compute complexity metrics, perform normality tests, and create publication-quality visualizations.

## Contents

- **`scripts/desc_stats.py`** - Main analysis script that:
  - Generates density field visualizations with velocity quiver plots
  - Computes Shannon entropy and complexity indices
  - Performs comprehensive normality tests (Shapiro-Wilk, Anderson-Darling, Jarque-Bera, D'Agostino K²)
  - Conducts inter-scenario statistical comparisons (Kruskal-Wallis, Mann-Whitney U)
  - Calculates temporal evolution metrics

- **`stats/`** - Generated statistical analysis outputs
- **`figs/`** - Generated figures (EPS, PDF, PNG formats)

## Requirements

```bash
numpy>=1.20.0
scipy>=1.7.0
xarray>=0.19.0
matplotlib>=3.4.0
```

## Usage

```bash
# Run analysis on simulation outputs
cd scripts
python desc_stats.py
```

The script expects NetCDF output files from kh2d-solver simulations in the `../outputs/` directory.

## Related Links

- **Main Solver**: [kh2d-solver on GitHub](https://github.com/sandyherho/kelvin-helmholtz-2d-solver)
- **PyPI Package**: [kh2d-solver](https://pypi.org/project/kh2d-solver/)
- **Simulation Data**: [OSF Archive](https://doi.org/10.17605/OSF.IO/HF6KX)

## Authors

- S.H.S. Herho
- N.J. Trilaksono 
- F.R. Fajary
- G. Napitupulu
- I.P. Anwar
- F. Khadami
- D.E. Irawan


## License

WTFPL - Do What The F*** You Want To Public License
