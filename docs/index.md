# Neuroscience Control

`neuroscience-control` is a PyTorch toolkit for whole-brain resting-state fMRI modeling.
It provides three model families with a shared training and evaluation pipeline:

- Coupled Hopf (physics-informed oscillators)
- Hybrid Hopf (Hopf + learnable coupling network)
- Neural SDE (data-driven stochastic dynamics)

## Why this package

- Complex-valued dynamical systems support for state and noise
- Reproducible training pipelines (`grid search`, `backprop`, `paper` run)
- Domain metrics out of the box: FC, FCD, phFCD, metastability, and timeseries fit
- Ready for package publishing to TestPyPI/PyPI and docs deployment via GitHub Pages

## Quick links

- Installation: [Getting Started / Installation](getting-started/installation.md)
- Tutorial: [First Training Run](tutorials/first-training-run.md)
- Metrics tutorial: [Metrics Evaluation](tutorials/metrics-evaluation.md)
- Publishing checklist: [Publishing](publishing.md)
