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

import time
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as data
from torchdiffeq import odeint
from tqdm.auto import tqdm
from .config import TokODEConfig
from .data import LinearInterpolator

logger = logging.getLogger(__name__)

def compute_huber_loss(pred: torch.Tensor, true: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    pred_li_ip = pred[..., :2]
    true_li_ip = true[..., :2]
    rel_error = torch.abs(pred_li_ip - true_li_ip) / (true_li_ip + 1e-8)
    huber_fn = nn.HuberLoss(delta=0.1, reduction='none')
    l_t = huber_fn(rel_error, torch.zeros_like(rel_error)).sum(dim=-1)
    dt = t[1:] - t[:-1]
    return torch.sum((dt / 2.0) * (l_t[1:] + l_t[:-1]))


class TokODEModelTrainer:
    def __init__(self, model: nn.Module, df: pd.DataFrame, config: TokODEConfig):
        self.model = model
        self.df = df
        self.config = config
        self.forward_method = 'rk4'
        self.torch_dtype = torch.float32
        self.model_type = self.model.__class__.__name__
        self.model_uid = getattr(self.model, 'uid', 'UNKNOWN_UID')

        unique_shots = self.df['shot'].unique()
        total_shots = len(unique_shots)

        if self.config.train_split >= total_shots:
            safe_split = max(1, int(0.8 * total_shots))
            logger.warning(
                f"[{self.model_uid}] train_split ({self.config.train_split}) >= total shots. "
                f"Forcing 80/20 split: {safe_split} train shots."
            )
        else:
            safe_split = self.config.train_split

        train_shots = unique_shots[:safe_split]
        val_shots = unique_shots[safe_split:]

        logger.info(
            f"[{self.model_uid}] Dataset Split -> Train: {len(train_shots)} shots | Val: {len(val_shots)} shots"
        )

        self.train_loader = data.DataLoader(
            self._build_dataset(train_shots), batch_size=self.config.batch_size, shuffle=True, collate_fn=lambda x: x
        )
        self.val_loader = data.DataLoader(
            self._build_dataset(val_shots), batch_size=self.config.batch_size, shuffle=False, collate_fn=lambda x: x
        )

        min_multiplier = self.config.min_learning_rate / self.config.learning_rate

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.config.learning_rate)
        
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda epoch: max(0.9 ** (epoch // 50), min_multiplier)
        )

        self.history = {
            "model_type": self.model_type,
            "model_uid": self.model_uid,
            "train_loss": [], 
            "val_loss": [],
            "li_mae": [], 
            "ip_mae": [], 
            "v_mae": [], 
            "li_pct": []
        }

    def _build_dataset(self, shots_list):
        cached = []
        for shot_id in shots_list:
            s_df = self.df[self.df['shot'] == shot_id].sort_values(
                'time').drop_duplicates(subset=['time'])
            t_vals = torch.tensor(
                s_df['time'].values - s_df['time'].iloc[0], dtype=self.torch_dtype)
            y0 = torch.tensor([s_df['li'].iloc[0], s_df['ip_MA'].iloc[0],
                              s_df['vc_minus_vb'].iloc[0]], dtype=self.torch_dtype)
            true_y = torch.tensor(
                s_df[['li', 'ip_MA', 'vc_minus_vb']].values, dtype=self.torch_dtype)
            interp = LinearInterpolator(
                pd.Series(t_vals.numpy()), s_df['Vind'])
            cached.append((t_vals, y0, true_y, interp))
        return cached
    def _compute_loss(self, pred: torch.Tensor, true: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Calculates the time-integrated Huber loss specifically for Li and Ip."""
        pred_li_ip = pred[..., :2]
        true_li_ip = true[..., :2]
        # relative error
        rel_error = torch.abs(pred_li_ip - true_li_ip) / (true_li_ip + 1e-8)
        huber_fn = nn.HuberLoss(delta=0.1, reduction='none') # Apply Huber (delta=0.1)
        l_t = huber_fn(rel_error, torch.zeros_like(rel_error)).sum(dim=-1)
        dt = t[1:] - t[:-1] # Integrate over time steps
        return torch.sum((dt / 2.0) * (l_t[1:] + l_t[:-1]))
    def _evaluate(self):
        self.model.eval()
        tot_loss, t_li_mae, t_ip_mae, t_v_mae, t_li_pct = 0, 0, 0, 0, 0
        n_shots = 0
        with torch.no_grad():
            for batch in self.val_loader:
                for t, y0, true_y, interp in batch:
                    self.model.interp = interp
                    pred = odeint(self.model, y0, t, method=self.forward_method, options={
                                  'step_size': self.config.solver_step})
                    tot_loss += compute_huber_loss(pred, true_y, t).item()
                    mae = torch.abs(pred - true_y).mean(dim=0)
                    t_li_mae += mae[0].item()
                    t_ip_mae += mae[1].item()
                    t_v_mae += mae[2].item()
                    t_li_pct += ((torch.abs(pred[..., 0] - true_y[..., 0]) / (
                        true_y[..., 0] + 1e-8)).mean() * 100).item()
                    n_shots += 1
        return (tot_loss/n_shots, (t_li_mae/n_shots, t_ip_mae/n_shots, t_v_mae/n_shots), t_li_pct/n_shots)

    def train(self):
        logger.info("="*65)
        logger.info(
            f"STARTING TRAINER: {self.model_type} | UID: {self.model_uid}")
        logger.info(
            f"Params: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
        logger.info("="*65)
        epoch_pbar = tqdm(range(self.config.num_epochs), desc=f"Training {self.model_type} Model")
        for epoch in epoch_pbar:
            start_t = time.time()
            self.model.train()
            ep_loss, n_shots = 0, 0

            for b_idx, batch in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                b_loss = 0
                for t, y0, true_y, interp in batch:
                    self.model.interp = interp
                    pred = odeint(
                        self.model, 
                        y0, 
                        t, 
                        method=self.forward_method, 
                        options={
                            'step_size': self.config.solver_step
                        }, 
                    )
                    b_loss += compute_huber_loss(pred, true_y, t)
                    n_shots += 1

                b_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                ep_loss += b_loss.item()

                if self.config.verbosity >= 1:
                    # Inject UID into debug logs
                    logger.debug(
                        f"[{self.model_uid}] Ep {epoch} | Batch {b_idx} | Loss: {b_loss.item():.5f}"
                    )

            self.scheduler.step()
            avg_train_loss = ep_loss / n_shots
            val_frequency = 1 if self.config.verbosity >= 1 else 10

            if epoch % val_frequency == 0 or epoch == self.config.num_epochs - 1:
                val_loss, maes, li_pct = self._evaluate()
                self.history["train_loss"].append(avg_train_loss)
                self.history["val_loss"].append(val_loss)
                self.history["li_mae"].append(maes[0])
                self.history["ip_mae"].append(maes[1])
                self.history["v_mae"].append(maes[2])
                self.history["li_pct"].append(li_pct)

                logger.info(
                    f"[{self.model_uid}] Ep: {epoch:4d} | Train: {avg_train_loss:.5f} | Val: {val_loss:.5f} | "
                    f"Li Error: {li_pct:.2f}% | Val MAE(V): {maes[2]:.4f} | Time: {time.time()-start_t:.2f}s"
                )
            epoch_pbar.set_postfix({'Avg Loss': f"{avg_train_loss:.6f}"})
        return self.history

    def report_performance(self):
        self.model.eval()
        li_errors, ip_errors, v_errors = [], [], []
        trajectories = []
        with torch.no_grad():
            for batch in self.val_loader:
                for t, y0, true_y, interp in batch:
                    self.model.interp = interp
                    pred = odeint(self.model, y0, t, method=self.forward_method, options={
                                  'step_size': self.config.solver_step})

                    eps = 1e-8
                    li_pct = (
                        torch.abs(pred[..., 0] - true_y[..., 0]) / (true_y[..., 0] + eps)) * 100
                    ip_pct = (
                        torch.abs(pred[..., 1] - true_y[..., 1]) / (true_y[..., 1] + eps)) * 100
                    v_pct = (torch.abs(
                        pred[..., 2] - true_y[..., 2]) / (torch.abs(true_y[..., 2]) + eps)) * 100

                    li_errors.append(li_pct.mean().item())
                    ip_errors.append(ip_pct.mean().item())
                    v_errors.append(v_pct.mean().item())

                    if len(trajectories) < 1:
                        trajectories.append(
                            {"t": t.cpu().numpy(), "true": true_y.cpu().numpy(), "pred": pred.cpu().numpy()})

        stats = {
            "model_type": self.model_type,
            "model_uid": self.model_uid,
            "li": {"mean": np.mean(li_errors), "std": np.std(li_errors)},
            "ip": {"mean": np.mean(ip_errors), "std": np.std(ip_errors)},
            "v":  {"mean": np.mean(v_errors),  "std": np.std(v_errors)}
        }

        logger.info("="*30 + " FINAL REPORT " + "="*30)
        logger.info(f"Model: {self.model_type} | UID: {self.model_uid}")
        logger.info(
            f"Li Percent Error: {stats['li']['mean']:.2f} ± {stats['li']['std']:.2f}%")
        logger.info(
            f"Ip Percent Error: {stats['ip']['mean']:.2f} ± {stats['ip']['std']:.2f}%")
        logger.info(
            f"V  Percent Error: {stats['v']['mean']:.2f}  ± {stats['v']['std']:.2f}%")
        logger.info("="*77)

        return stats, trajectories
