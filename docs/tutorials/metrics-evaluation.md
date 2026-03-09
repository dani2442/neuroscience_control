# Metrics Evaluation

The package includes static FC, dynamics, and timeseries metrics.

## Example: FC + dynamics metrics

```python
import torch
from neuroscience_control.metrics import compute_all_fc_metrics, compute_dynamics_fit_metrics

# sim_ts and real_ts: [batch, n_rois, n_timepoints], complex-valued
sim_fc = torch.corrcoef(sim_ts.real[0])
real_fc = torch.corrcoef(real_ts.real[0])

fc_metrics = compute_all_fc_metrics(sim_fc.unsqueeze(0), real_fc.unsqueeze(0))
dyn_metrics = compute_dynamics_fit_metrics(
    sim_ts,
    real_ts,
    tr=0.72,
    fcd_win_sec=30.0,
    fcd_step_sec=2.0,
)

print(fc_metrics)
print(dyn_metrics)
```

## Key metrics used in training/evaluation

- `fc_correlation`
- `fc_mse`
- `fcd_ks`
- `phfcd_ks`
- `metastability_diff`
- `temporal_correlation`
- `power_spectrum_distance`
- `autocorr_distance`

These metrics are aggregated in `neuroscience_control.utils.EVAL_METRIC_KEYS`.
