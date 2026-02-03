import numpy as np
import os
from scipy.io import loadmat

data = loadmat("data/ts_young/ts_young_TR0.72.mat")

data['FC_all'].shape # (number of ROIs, number of ROIs, number of subjects)
data['FC_mean'].shape # (number of ROIs, number of ROIs)
data['timeseries_all'].shape # (number of ROIs, number of timepoints, number of subjects)