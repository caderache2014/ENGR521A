# Copyright 2026 The tokode-torch Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
models.py
PyTorch model definitions and factory enumeration for tokamak ODEs.
"""

import os
import uuid
import logging
from enum import Enum
from typing import Optional, Union

import pandas as pd
import torch
import torch.nn as nn
from torchdiffeq import odeint

from ._types.types import ODESolverLiteral, SupportedODESolverMethods
from .utils.data import calculate_df_norm

logger = logging.getLogger(__name__)

class ShotInterpolator(nn.Module):
    """
    Wraps discrete time-series control data into a continuous function.
    Vectorized to support both scalar and tensor inputs.
    """
    def __init__(self, t_data: torch.Tensor, v_data: torch.Tensor):
        super().__init__()
        self.register_buffer('t_data', t_data)
        self.register_buffer('v_data', v_data)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_clamped = torch.clamp(t, min=self.t_data[0], max=self.t_data[-1])
        idx = torch.searchsorted(self.t_data, t_clamped)
        idx = torch.clamp(idx, min=1, max=len(self.t_data) - 1)
        t0, t1 = self.t_data[idx - 1], self.t_data[idx]
        v0, v1 = self.v_data[idx - 1], self.v_data[idx]
        weight = (t_clamped - t0) / (t1 - t0)
        v_interp = v0 + weight * (v1 - v0)
        return v_interp
    
class TokODEModel(nn.Module):
    """
    Base class for all Tokamak ODE models.
    Handles saving, loading, and generating simulated trajectories.
    """
    def __init__(self):
        super().__init__()
        self.uid = uuid.uuid4()
        logger.info(f"[{self.__class__.__name__}] Initialized with UID: {self.uid}")

    def forward(self, t: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """Must be implemented by child classes to define the ODE system."""
        raise NotImplementedError("Child classes must implement the ODE forward pass.")

    def export_model(self, save_dir: Optional[str] = None):
        """Saves model with its unique ID included in the filename, with error handling.
        Defaults to the system /tmp/ directory if no path is provided.
        """
        try:
            save_dir = os.path.join(os.getcwd(), "tmp/models") if not save_dir else save_dir
            os.makedirs(save_dir, exist_ok=True)
            model_type = self.__class__.__name__.lower()
            file_name = f"{model_type}_{self.uid}.pth"
            file_path = os.path.join(save_dir, file_name)
            torch.save(self.state_dict(), file_path)
            logger.info(f"Model successfully exported to {file_path}")
            return file_path

        except PermissionError:
            raise PermissionError(f"Critical: Permission denied when writing to {save_dir}. Check folder permissions.")
        except OSError as e:
            raise OSError(f"Critical: OS error occurred while saving to {save_dir}: {e}")
        except Exception as e:
            raise RuntimeError(f"An unexpected error occurred during export: {e}")

    def load_model(self, model_path: str):
        """Loads weights from a saved state dict with error handling for missing files or corruption."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Load Error: The file {model_path} does not exist.")
            
        try:
            # Attempt to load and map to current device
            state_dict = torch.load(model_path, map_location=torch.device('cpu'), weights_only=True)
            self.load_state_dict(state_dict, strict=False)
            self.eval() 
            logger.info(f"Model weights loaded from {model_path} and locked in eval mode.")
            
        except RuntimeError as e:
            raise RuntimeError(f"Load Error: Failed to load state_dict. Model architecture might mismatch: {e}")
        except Exception as e:
            raise IOError(f"Load Error: An error occurred while reading the file: {e}")

    def predict_trajectory(
        self, 
        t_eval: torch.Tensor, 
        initial_state: torch.Tensor, 
        solver_method: ODESolverLiteral = 'dopri5',
        rtol: float = 1e-5,
        atol: float = 1e-5
    ) -> torch.Tensor:
        """
        Simulates the ODE forward in time to generate the predicted trajectory.
        Includes an automatic fallback to 'rk4' if adaptive solvers underflow.
        """
        # Runtime validation check using our Enum
        if solver_method not in SupportedODESolverMethods.list_values():
            raise ValueError(
                f"Unsupported solver: '{solver_method}'. "
                f"Supported methods are: {SupportedODESolverMethods.list_values()}"
            )

        self.eval()
        
        with torch.no_grad():
            try:
                # Attempt the primary solver requested by the user
                predicted_trajectory = odeint(
                    self, 
                    initial_state, 
                    t_eval, 
                    method=solver_method,
                    rtol=rtol,
                    atol=atol
                )
            except AssertionError as e:
                # Check if the error is our specific dt underflow
                if 'underflow in dt' in str(e).lower() and solver_method != SupportedODESolverMethods.RK4.value:
                    logger.warning(f"'{solver_method}' failed with dt underflow. Falling back to 'rk4'...")
                    
                    # Execute the fallback fixed-step solver
                    predicted_trajectory = odeint(
                        self, 
                        initial_state, 
                        t_eval, 
                        method=SupportedODESolverMethods.RK4.value
                    )
                else:
                    # If it's a completely different error, show it
                    raise e
                    
        return predicted_trajectory

class RomeroNNV(TokODEModel):
    """Hybrid model replacing voltage dynamics with a neural network."""

    def __init__(
        self,
        interpolator: Optional[nn.Module] = None,
        romero_norm: float = 0.85
    ):
        super().__init__()
        self.interp = interpolator
        self.romero_norm = romero_norm
        self.mlp = nn.Sequential(
            nn.Linear(4, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )

    def forward(self, t: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        li, ip_ma, v = state[..., 0], state[..., 1], state[..., 2]

        vb_minus_vr = self.interp(t) if self.interp is not None else torch.zeros_like(v)
        vr_minus_vb = -vb_minus_vr
        denom_ip = self.romero_norm * ip_ma + 1e-8
        denom_li = self.romero_norm * li + 1e-8
        
        li_dot = (2 * vr_minus_vb - 2 * v) / denom_ip
        ip_ma_dot = (2 * vb_minus_vr + v) / denom_li

        nn_input = torch.stack([li, ip_ma, v, vb_minus_vr], dim=-1)
        v_dot = self.mlp(nn_input).squeeze(-1)

        return torch.stack([li_dot, ip_ma_dot, v_dot], dim=-1)


class RomeroModel(TokODEModel):
    """Pure physics baseline model based on Romero (2010)."""

    def __init__(
        self,
        interpolator: Optional[nn.Module] = None,
        romero_norm: float = 0.85
    ):
        super().__init__()
        self.interp = interpolator
        self.romero_norm = romero_norm
        self.kappa = nn.Parameter(torch.tensor(0.98))
        self.tau = nn.Parameter(torch.tensor(1.25))

    def forward(self, t: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        li, ip_ma, v = state[..., 0], state[..., 1], state[..., 2]
        
        vb_minus_vr = self.interp(t) if self.interp is not None else torch.zeros_like(v)
        vr_minus_vb = -vb_minus_vr

        denom_ip = self.romero_norm * ip_ma + 1e-8
        denom_li = self.romero_norm * li + 1e-8
        
        li_dot = (2 * vr_minus_vb - 2 * v) / denom_ip
        ip_ma_dot = (2 * vb_minus_vr + v) / denom_li
        v_dot = -(v / self.tau) + (self.kappa / self.tau) * vr_minus_vb
        
        return torch.stack([li_dot, ip_ma_dot, v_dot], dim=-1)


class MlpODE(TokODEModel):
    """Pure neural network baseline ignoring physical equations."""

    def __init__(self, interpolator: Optional[nn.Module] = None):
        super().__init__()
        self.interp = interpolator
        self.mlp = nn.Sequential(
            nn.Linear(4, 32),
            nn.Tanh(),
            nn.Linear(32, 3) 
        )

    def forward(self, t: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        li, ip_ma, v = state[..., 0], state[..., 1], state[..., 2]
        
        vind = self.interp(t) if self.interp is not None else torch.zeros_like(v)
        nn_input = torch.stack([li, ip_ma, v, vind], dim=-1)
        
        return self.mlp(nn_input)


class TokODEModelType(Enum):
    """Enumeration factory for model instantiation."""
    HYBRID = "RomeroNNV"
    PHYSICS = "RomeroModel"
    PURE_AI = "MlpODE"

    def build(self, norm: Union[float, str] = 0.85, df: pd.DataFrame = None) -> TokODEModel:
        """Instantiates the specified model type.
        If norm='auto', it calculates the exact physical constant from the provided dataframe.
        """
        if norm == 'auto':
            if df is None:
                raise ValueError("You must provide a dataset (df) to calculate norm='auto'")
            resolved_norm = calculate_df_norm(df=df)
            print(f"Auto-calculated norm from dataset: {norm:.4f}")
        elif isinstance(norm, float) or isinstance(norm, int):
            resolved_norm = float(norm)
        else:
            raise ValueError("Norm must be a float or the string 'auto'")
        if self == TokODEModelType.HYBRID:
            return RomeroNNV(romero_norm=resolved_norm)
        elif self == TokODEModelType.PHYSICS:
            return RomeroModel(romero_norm=resolved_norm)
        elif self == TokODEModelType.PURE_AI:
            return MlpODE()
        else:
            raise ValueError(f"No build logic defined for {self.value}")
