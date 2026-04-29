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

import os
from pathlib import Path
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def fetch_data(filepath: os.PathLike, compression: str | None = None) -> pd.DataFrame:
    """Loads and sorts the tokamak dataset."""
    if not Path(filepath).is_file():
        raise FileNotFoundError(f"Dataset not found at: {filepath}. Check your relative path.")
        
    if filepath.endswith('.zip'):
        compression = 'zip'
    df = pd.read_csv(filepath, compression=compression)
    return df.sort_values(by=['shot', 'time']).reset_index(drop=True)

def calculate_df_norm(df: pd.DataFrame) -> float:
    """Calculates the empirical geometric norm from the dataset."""
    logger.info("Scanning dataset to calculate empirical geometric norm...")
    
    norms = []
    for shot_id, group in df.groupby('shot'):
        group = group.sort_values('time')
        dt = group['time'].diff()
        dLi_dt = group['li'].diff() / dt
        
        numerator = -2 * group['Vind'].shift(1) - 2 * group['vc_minus_vb'].shift(1)
        denominator = group['ip_MA'].shift(1) * dLi_dt
        
        valid_mask = np.abs(denominator) > 1e-5
        shot_norms = (numerator[valid_mask] / denominator[valid_mask]).dropna()
        norms.extend(shot_norms.values)

    # Calculate the final median norm
    final_norm = float(np.median(norms))
    
    # Print and Log the result
    print(f"Auto-calculated empirical norm: {final_norm:.4f}")
    logger.info(f"Auto-calculated empirical norm: {final_norm:.4f}")

    return final_norm