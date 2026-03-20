# API Overview

The current import namespace in this repository is `src`.

## Core modules

- `src.dataset`
- `src.models`
- `src.training`
- `src.metrics`
- `src.utils`

## Common imports

```python
from src.dataset import NeuroscienceDataset, create_data_loaders, load_dataset
from src.models import (
    CoupledHopfModel,
    HybridHopfModel,
    GNNHopfModel,
    NeuralSDE,
    build_model,
    load_model_from_checkpoint,
)
from src.training import (
    Trainer,
    GridSearch,
    grid_search_hopf,
    TrainingConfig,
    HopfConfig,
    HybridHopfConfig,
    GNNHopfConfig,
    NeuralSDEConfig,
    CompositeLoss,
)
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
    L2Timeseries,
    AmplitudeLoss,
    OmegaLoss,
    MetricsStore,
)
from src.utils import EVAL_METRIC_KEYS, evaluate_model_loader_metrics
```

## Module map

### `src.dataset`

- Dataset constructors and backend loading.
- Random window sampling and split helpers.
- Frequency estimation and signal preprocessing helpers.

### `src.models`

- Four model classes with a shared `forward()` signature.
- A model factory used by the example CLIs.
- Checkpoint loading that reconstructs architectures from saved metadata.

### `src.training`

- Config dataclasses for each training family.
- `Trainer` for backprop-based optimization.
- `GridSearch` / `grid_search_hopf` for Hopf parameter sweeps.
- `CompositeLoss` for assembling weighted training objectives.

### `src.metrics`

- FC, dynamics, and timeseries metrics as reusable `nn.Module` objects.
- Reference-stat helpers for amplitude and intrinsic frequency.
- `MetricsStore` for JSON-backed metric accumulation.

### `src.utils`

- Evaluation helpers for loader-level metric computation.
- Plotting utilities and figure generation.
- Runtime helpers for device selection, seeding, and W&B integration.
