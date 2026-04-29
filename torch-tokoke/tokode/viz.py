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
from typing import List, Optional, Union, List, Dict
from IPython.display import display, Markdown
import matplotlib.pyplot as plt
import numpy as np
from ._types.types import ExperimentResult

def display_results_table(results: Union[Dict, List[Dict]], precision: int = 4) -> None:
    """
    Creates and renders a formatted Markdown table of model evaluation results.
    
    Args:
        results: A single dictionary or a list of dictionaries containing model stats.
        precision: Number of decimal places to format the float values.
    """
    # Normalize input to a list so we can handle both single dicts and lists of dicts
    if isinstance(results, dict):
        results = [results]

    # Define the table headers
    headers = [
        "Model Type", 
        "Model UID", 
        "$L_i$ (Mean ± Std)", 
        "$I_p$ (Mean ± Std)", 
        "$V$ (Mean ± Std)"
    ]
    
    # Build the Markdown table structure
    md_lines = []
    md_lines.append("| " + " | ".join(headers) + " |")
    md_lines.append("|" + "|".join([":---" for _ in headers]) + "|")

    for res in results:
        m_type = res.get('model_type', 'Unknown')

        uid_full = str(res.get('model_uid', 'N/A'))
        uid_short = uid_full.split('-')[0] if '-' in uid_full else uid_full[:8]
        li_str = f"{res['li']['mean']:.{precision}f} ± {res['li']['std']:.{precision}f}"
        ip_str = f"{res['ip']['mean']:.{precision}f} ± {res['ip']['std']:.{precision}f}"
        v_str  = f"{res['v']['mean']:.{precision}f} ± {res['v']['std']:.{precision}f}"
        
        row = f"| **{m_type}** | `{uid_short}` | {li_str} | {ip_str} | {v_str} |"
        md_lines.append(row)
    display(Markdown("\n".join(md_lines)))

def plot_multi_model_results(results: List[ExperimentResult], save_path: str = None):
    """Plots validation loss and predictive trajectories overlapping on the same graphs."""
    color_map = {
        "RomeroNNV": "darkorange",  
        "RomeroModel": "deeppink",  
        "MlpODE": "#3CB371"         
    }

    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Model Comparison: Validation Loss & Trajectories', fontsize=18, fontweight='bold')

    ax_loss, ax_li, ax_ip, ax_v = axs[0, 0], axs[0, 1], axs[1, 0], axs[1, 1]

    ax_loss.set(yscale='log', title="Validation Loss (Huber)", xlabel="Epoch Index", ylabel="Loss")
    ax_li.set(title="Internal Inductance ($L_i$)", xlabel="Time (s)", ylabel="$L_i$")
    ax_ip.set(title="Plasma Current ($I_p$)", xlabel="Time (s)", ylabel="$I_p$ (MA)")
    ax_v.set(title="Voltage ($V$)", xlabel="Time (s)", ylabel="$V$")

    for ax in [ax_loss, ax_li, ax_ip, ax_v]:
        ax.grid(True, alpha=0.3)

    true_data_plotted = False

    for res in results:
        m_name = res.model_name
        color = color_map.get(m_name, "blue") 

        if "val_loss" in res.history and res.history["val_loss"]:
            ax_loss.plot(res.history["val_loss"], label=m_name, color=color, linewidth=2)

        if res.plot_data:
            sample = res.plot_data[0]
            t, true_y, pred_y = sample["t"], sample["true"], sample["pred"]

            if not true_data_plotted:
                ax_li.plot(t, true_y[:, 0], 'k--', label="True Data", linewidth=2.5, zorder=10)
                ax_ip.plot(t, true_y[:, 1], 'k--', label="True Data", linewidth=2.5, zorder=10)
                ax_v.plot(t, true_y[:, 2], 'k--', label="True Data", linewidth=2.5, zorder=10)
                true_data_plotted = True

            ax_li.plot(t, pred_y[:, 0], color=color, label=m_name, linewidth=2, alpha=0.8)
            ax_ip.plot(t, pred_y[:, 1], color=color, label=m_name, linewidth=2, alpha=0.8)
            ax_v.plot(t, pred_y[:, 2], color=color, label=m_name, linewidth=2, alpha=0.8)

    ax_loss.legend()
    ax_li.legend()
    ax_ip.legend()
    ax_v.legend()

    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    else:
        plt.show()

        """Visualization utilities for Tokamak ODE experiment results."""


def plot_single_run(
    plot_data: List[dict], 
    model_name: str = "Model", 
    save_path: Optional[str] = None
):
    """Plots true vs predicted trajectories for a single model run.

    Args:
        plot_data: A list of dictionaries containing 't', 'true', and 'pred' arrays.
        model_name: String name of the model for the plot title.
        save_path: Optional path to save the figure.
    """
    # We typically plot the first shot in the batch for visualization
    item = plot_data[0]
    t = item['t']
    true = item['true']
    pred = item['pred']

    fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    labels = [r'$L_i$', r'$I_p$', r'$V$']
    ylabels = [r'$L_i$', r'$I_p$ (MA)', r'$V$ (V)']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Professional color palette

    for i in range(3):
        axs[i].plot(t, true[:, i], 'k--', alpha=0.7, label='Experimental True')
        axs[i].plot(t, pred[:, i], color=colors[i], linewidth=2, label=f'Pred ({model_name})')
        axs[i].set_ylabel(ylabels[i])
        axs[i].grid(True, linestyle=':', alpha=0.6)
        axs[i].legend(loc='upper right')

    axs[2].set_xlabel('Time (s)')
    fig.suptitle(f'State Trajectory Reconstruction: {model_name}', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Plot saved to {save_path}")
    
    plt.show()


def plot_multi_model_comparison(
    all_results: List['ExperimentResult'], 
    save_path: Optional[str] = None
):
    """Plots training loss and final state error for multiple models.

    Args:
        all_results: List of ExperimentResult objects from the pipeline.
        save_path: Optional path to save the comparison figure.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # 1. Training Loss Comparison
    for res in all_results:
        epochs = range(1, len(res.history) + 1)
        ax1.plot(epochs, res.history, label=res.model_name, linewidth=2)

    ax1.set_yscale('log')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss (MSE)')
    ax1.set_title('Training Convergence')
    ax1.legend()
    ax1.grid(True, which="both", ls="-", alpha=0.2)

    # 2. Final Error Comparison (Boxplot or Bar)
    model_names = [res.model_name for res in all_results]
    
    # Calculate Mean Squared Error across the visualized shot
    final_errors = []
    for res in all_results:
        true, pred = res.plot_data[0]['true'], res.plot_data[0]['pred']
        mse = np.mean((true - pred)**2)
        final_errors.append(mse)

    bars = ax2.bar(model_names, final_errors, color=['gray', 'blue', 'orange'])
    ax2.set_ylabel('Mean Squared Error')
    ax2.set_title('Final Reconstruction Accuracy')
    
    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.4f}', ha='center', va='bottom')

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
    
    plt.show()