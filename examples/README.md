# Examples

Reproducibility scripts and notebooks for the figures in `paper_new/`.

## Entry-point scripts

| Script | Purpose |
|---|---|
| [`train_models.py`](train_models.py) | Train one or all model variants. `paper` subcommand runs the full training suite. |
| [`postprocess.py`](postprocess.py) | Update tables, generate comparison figures, run condition sweeps. `pipeline` subcommand runs the full post-training pipeline. |
| [`benchmark_timing.py`](benchmark_timing.py) | Wall-clock benchmarks across model classes. |
| [`cli_args.py`](cli_args.py) | Shared argparse helpers used by the above. |

End-to-end run for the paper figures:

```bash
# Train all models (HCP / Young dataset)
python examples/train_models.py paper \
    --dataset-type ts_young --data-path data/ts_young/ts_young_TR0.72.mat

# Post-training: update tables + comparison figures
python examples/postprocess.py pipeline \
    --dataset-type ts_young --data-path data/ts_young/ts_young_TR0.72.mat
```

## Notebooks → paper figures

Notebooks are prefixed by the paper figure they generate panels for.

| Notebook | Paper figure / panel |
|---|---|
| [`fig1_data_pipeline.ipynb`](fig1_data_pipeline.ipynb) | Fig. 1A — BOLD → bandpass → analytic-signal pipeline (`images/representation/` panels embedded in the Fig. 1 drawio) |
| [`fig1_brain_renderings.ipynb`](fig1_brain_renderings.ipynb) | Fig. 1C — empirical vs. simulated brain at fixed timestep (`wholebrain_timestep_lateral_magma`) |
| [`fig2_grid_vs_gradient.ipynb`](fig2_grid_vs_gradient.ipynb) | Fig. 2C — grid-search vs. gradient-descent metric bars across seeds |
| [`fig3_fc_fcd_phfcd.ipynb`](fig3_fc_fcd_phfcd.ipynb) | Fig. 3B(A) — per-model FC / FCD / phFCD matrices (`wholebrain_matrices/`) |
| [`fig3_training_curves.ipynb`](fig3_training_curves.ipynb) | Fig. 3B(B) — training loss + validation metrics over epochs |
| [`fig3_clustering_complexity.ipynb`](fig3_clustering_complexity.ipynb) | Fig. 3B(C) brain clustering + Fig. 3B(F) entropy / LZW complexity |
| [`fig3_personalization.ipynb`](fig3_personalization.ipynb) | Fig. 3B(D) — intra- vs. inter-subject FC reconstruction bars |
| [`fig3_size_robustness.ipynb`](fig3_size_robustness.ipynb) | Fig. 3B(E) — dataset-size sweep |
| [`fig3_brain_panels.ipynb`](fig3_brain_panels.ipynb) | Fig. 3B(A) brain-rendering side panels (`wholebrain/wholebrain_*_panel.png`) |

The four PNGs actually referenced from `paper_new/main.tex` live in `paper_new/images/diagrams/` and are exported from drawio sources in the same folder. Each drawio embeds the notebook-generated panels as base64, so the drawios are self-contained but the notebooks here remain the canonical source of truth for regenerating any panel.

## Running notebooks

All notebooks expect to be executed from the repo root or from `examples/`. They self-locate the project root:

```python
project_root = Path.cwd()
if not (project_root / 'src').exists():
    project_root = project_root.parent
```

If you move a notebook elsewhere, update that snippet.
