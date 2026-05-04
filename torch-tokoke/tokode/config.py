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
config.py
Defines the configuration inputs for the Tokamak ODE training pipeline.
"""

from pydantic import BaseModel, Field
from ._types.types import TokODELoggingVerbosityLevel


class TokODEConfig(BaseModel):
    """Configuration mimicking SFTConfig for Tokamak ODE training."""

    batch_size: int = Field(
        default=512,
        description="Batch size for training."
    )
    train_split: int = Field(
        default=50,
        description="Number of shots allocated for training."
    )
    num_epochs: int = Field(
        default=10000,
        description="Total maximum epochs to train."
    )
    learning_rate: float = Field(
        default=1e-3,
        description="Initial learning rate."
    )
    min_learning_rate: float = Field(
        default=2.5e-4,
        description="Floor limit for learning rate decay."
    )
    solver_step: float = Field(
        default=0.01,
        description="Fixed step size for the ODE solver."
    )
    verbosity: TokODELoggingVerbosityLevel = Field(
        default=TokODELoggingVerbosityLevel.NORMAL,
        description="Controls the detail level of console output."
    )
