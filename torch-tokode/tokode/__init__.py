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
TokODE ODE
A Neural ODE pipeline for modeling internal inductance, plasma current, 
and voltage dynamics in tokamak plasmas.
"""

__version__ = "0.1.0"

# from .config import TokODEConfig
from .config import *
from .viz import display_results_table, plot_multi_model_results
from .utils.data import fetch_data, calculate_df_norm

from ._types.types import (
    TokODELoggingVerbosityLevel,
    SupportedODESolverMethods,
    ModelArchitecture,
    ExecutionPhase,
    TokODEShotData,
    ExperimentResult
)

from .models import (
    TokODEModel,
    TokODEModelType,
    RomeroNNV,
    MlpODE,
    RomeroModel,
    ShotInterpolator
)

from .data import LinearInterpolator, CubicSplineInterpolator

from .trainer import TokODEModelTrainer
__all__ = [
    "TokODEConfig",
    "TokODELoggingVerbosityLevel",
    "SupportedODESolverMethods",
    "ModelArchitecture",
    "ExecutionPhase",
    "TokODEShotData",
    "ExperimentResult",
    "TokODEModel",
    "TokODEModelType",
    "RomeroNNV",
    "MlpODE",
    "RomeroModel",
    "ShotInterpolator",
    "TokODEModelTrainer", 
    "LinearInterpolator",
    "CubicSplineInterpolator"
]