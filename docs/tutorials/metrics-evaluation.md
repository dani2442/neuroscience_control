# Metrics Evaluation

Metrics are implemented as `nn.Module` classes under `src.metrics`. Each metric accepts complex analytic signals with shape `(batch, n_rois, T)`.

- `forward(ts_pred, ts_target)` returns a differentiable scalar loss.
- `evaluate(ts_pred, ts_target)` returns one or more logging-ready metric values.

## Example: evaluate a simulated batch

```python
from src.metrics import (
    FCCorrelation,
    FCMSE,
    FCD,
    PhFCD,
    Metastability,
    PhaseFC,
    PowerSpectrumDistance,
    TemporalCorrelation,
    AutocorrelationDistance,
)

modules = [
    FCCorrelation(),
    FCMSE(),
    FCD(tr=0.72, fcd_win_sec=30.0, fcd_step_sec=2.0),
    PhFCD(),
    Metastability(),
    PhaseFC(),
    PowerSpectrumDistance(),
    TemporalCorrelation(),
    AutocorrelationDistance(),
]

metrics = {}
for module in modules:
    metrics.update(module.evaluate(sim_ts, real_ts))

print(metrics)
```

## Metric groups

- FC: `FCCorrelation`, `FCMSE`
- Dynamics / phase: `FCD`, `PhFCD`, `Metastability`, `PhaseFC`
- Timeseries: `PowerSpectrumDistance`, `TemporalCorrelation`, `AutocorrelationDistance`
- Auxiliary training losses: `L2Timeseries`, `AmplitudeLoss`, `OmegaLoss`

FC, spectrum, and autocorrelation metrics operate on the real part of the analytic signal. Phase-based metrics derive phases with `torch.angle(...)`.

## Composite training loss

`src.training.CompositeLoss` wires the same modules into a weighted training objective:

```python
from src.training import CompositeLoss

loss_fn = CompositeLoss(
    weights={
        "fc_correlation": 1.0,
        "phfcd": 1.0,
        "metastability": 1.0,
    },
    tr=0.72,
    fcd_win_sec=30.0,
    fcd_step_sec=2.0,
)

total_loss, components = loss_fn(sim_ts, real_ts)
print(total_loss)
print(components)
```

## Loader-level evaluation

For end-to-end model evaluation on a `DataLoader`, use `src.utils.evaluate_model_loader_metrics`:

```python
from src.utils import EVAL_METRIC_KEYS, evaluate_model_loader_metrics

metrics = evaluate_model_loader_metrics(model, val_loader, cfg, return_std=True)

for key in EVAL_METRIC_KEYS:
    print(key, metrics.get(key), metrics.get(f"{key}_std"))
```

The default report keys are:

- `fc_correlation`
- `fc_mse`
- `temporal_correlation`
- `power_spectrum_distance`
- `autocorr_distance`
- `fcd_ks`
- `phfcd_ks`
- `phase_fc_correlation`
- `metastability_diff`
