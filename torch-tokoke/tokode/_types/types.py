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

from enum import Enum, IntEnum
from typing import Literal, Dict, List, Any
from pydantic import BaseModel, Field

ODESolverLiteral = Literal["dopri5", "rk4", "euler", "midpoint"]

################################################################################
from typing import Union, List, Tuple
import torch
import numpy as np
import pandas as pd

# For utilities that can ingest multiple formats before converting to a Tensor
ArrayLike = Union[torch.Tensor, np.ndarray, pd.Series, List[float]]

# Semantic aliases to make function signatures self-documenting
TimeGrid = torch.Tensor       # Expected shape: (T,)
StateVector = torch.Tensor    # Expected shape: (Batch, State_Dim)
Trajectory = torch.Tensor     # Expected shape: (Batch, T, State_Dim)

class SupportedODESolverMethods(Enum):
    DOPRI5 = "dopri5"
    RK4 = "rk4"
    EULER = "euler"
    MIDPOINT = "midpoint"

    @classmethod
    def list_values(cls):
        return [method.value for method in cls]

class TokODELoggingVerbosityLevel(IntEnum):
    NORMAL = 0
    VERBOSE = 1
    DEBUG = 2

class ExperimentResult(BaseModel):
    """Structure to hold the final outputs of a training run."""
    model_type: str
    model_uid: str
    history: Dict[str, Any]
    stats: Dict[str, Any]
    trajectories: List[Dict[str, Any]] = Field(default_factory=list)



class TokODEShotData(BaseModel):
    """Represents a single, validated experimental shot."""
    shot_id: int
    time: ArrayLike = Field(description="Time vector in seconds")
    li: ArrayLike = Field(description="Internal Inductance")
    ip: ArrayLike = Field(description="Plasma Current in MA")
    v: ArrayLike = Field(description="Voltage (vc_minus_vb)")
    v_ind: ArrayLike = Field(description="Inductive Control Voltage")
    
    class Config:
        arbitrary_types_allowed = True



class ModelArchitecture(str, Enum):
    """Supported neural architectures."""
    HYBRID_NNV = "RomeroNNV"
    PURE_MLP = "MlpODE"
    PHYSICS_ONLY = "RomeroModel"

class ExecutionPhase(str, Enum):
    """Tracks the current state of the pipeline."""
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"
    INFERENCE = "inference"