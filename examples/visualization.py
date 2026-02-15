import torch
from pathlib import Path
import sys
import matplotlib.pyplot as plt

# Ensure imports work when running this file directly (absolute or relative path).
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models import CoupledHopfModel, NeuralSDE
from src.dataset import NeuroscienceDataset, compute_omega_from_timeseries


dataset = NeuroscienceDataset(filepath="data/ts_young/ts_young_TR0.72.mat", normalize=True, max_subjects=2, fourier_denoise=True)
omega = compute_omega_from_timeseries(dataset.timeseries, dt=dataset.dt, f_lo=0.04, f_hi=0.07, method="peak")

patient = 0
n_steps = 40

model = CoupledHopfModel(
    n_rois=100,
    initial_a=-0.02,
    initial_g=1.5, 
    initial_kappa=0.1,
    noise_sigma=0.2,
    omega=omega,
    structural_connectivity=dataset.fc_mean
)

# Simulate 200 timepoints
with torch.no_grad():
    z0 = dataset.timeseries[:, :, 0]  # complex initial conditions
    zs = model.forward(initial_state=z0, n_steps=n_steps)  # (10, 68, 200) complex
    fc_matrix = model.compute_fc(zs)     


fig, ax = plt.subplots(figsize=(5, 5))

ax.plot(zs[patient, 0].real, zs[patient, 0].imag, label="Simulation")
ax.plot(dataset.timeseries[patient, 0, :n_steps].real, dataset.timeseries[patient, 0, :n_steps].imag, label="Dataset", marker='x')
    # axes[i].set_xticklabels([])
    # axes[i].set_yticklabels([])
    # axes[i].set_xlim(-1, 1)
    # axes[i].set_ylim(-1, 1)

plt.tight_layout()
plt.show()