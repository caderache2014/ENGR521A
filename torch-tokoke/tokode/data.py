"""
data.py
Data loading, preprocessing, and continuous interpolation utilities.
"""

import os
from pathlib import Path
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from typing import Union
from scipy.interpolate import CubicSpline

class LinearInterpolator(nn.Module):
    """Interpolates control signals (Vind) continuously over time for the ODE solver."""
    def __init__(self, t_data: pd.Series, y_data: pd.Series):
        super().__init__()
        self.register_buffer('t', torch.as_tensor(t_data.values.copy(), dtype=torch.float32))
        self.register_buffer('y', torch.as_tensor(y_data.values.copy(), dtype=torch.float32))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t = torch.clamp(t, self.t[0], self.t[-1])
        idx = torch.clamp(torch.searchsorted(self.t, t), 1, len(self.t) - 1)
        t0, t1 = self.t[idx-1], self.t[idx]
        y0, y1 = self.y[idx-1], self.y[idx]
        return y0 + (y1 - y0) * (t - t0) / (t1 - t0)
    
class CubicSplineInterpolator(nn.Module):
    """
    Wraps discrete time-series control data into a continuous function
    using precomputed Cubic Splines for perfectly smooth derivatives.
    Vectorized and GPU-compatible for ODE solvers.
    """
    def __init__(self, t_data: Union[torch.Tensor, pd.Series, np.ndarray], v_data: Union[torch.Tensor, pd.Series, np.ndarray]):
        super().__init__()
        t_np = t_data.values if isinstance(t_data, pd.Series) else np.array(t_data)
        v_np = v_data.values if isinstance(v_data, pd.Series) else np.array(v_data)
        t_np, v_np = t_np.flatten(), v_np.flatten()
        cs = CubicSpline(t_np, v_np)
        coeffs = torch.tensor(cs.c, dtype=torch.float32)
        t_tensor = torch.tensor(t_np, dtype=torch.float32)
        self.register_buffer('t_data', t_tensor)
        self.register_buffer('coeffs', coeffs)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Evaluates the precomputed cubic spline at continuous time step `t`."""
        t_clamped = torch.clamp(t, min=self.t_data[0], max=self.t_data[-1])
        idx = torch.searchsorted(self.t_data, t_clamped)
        idx = torch.clamp(idx - 1, min=0, max=self.t_data.shape[0] - 2)
        dt = t_clamped - self.t_data[idx]
        a = self.coeffs[0, idx]
        b = self.coeffs[1, idx]
        c = self.coeffs[2, idx]
        d = self.coeffs[3, idx]
        # Evaluate polynomial: a*dt^3 + b*dt^2 + c*dt + d
        return a * (dt**3) + b * (dt**2) + c * dt + d