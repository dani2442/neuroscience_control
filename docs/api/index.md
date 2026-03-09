# API Overview

Current public modules:

- `neuroscience_control.dataset`
- `neuroscience_control.models`
- `neuroscience_control.training`
- `neuroscience_control.metrics`
- `neuroscience_control.utils`

## Import examples

```python
from neuroscience_control.models import CoupledHopfModel, HybridHopfModel, NeuralSDE
from neuroscience_control.training import HopfConfig, NeuralSDEConfig, run_backprop_training
from neuroscience_control.metrics import compute_all_fc_metrics, compute_dynamics_fit_metrics
```

## Backwards compatibility

Legacy imports continue to work:

```python
from src.models import CoupledHopfModel
```

Use `neuroscience_control.*` in new code.
