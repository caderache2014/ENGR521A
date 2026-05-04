# **`ENGR 521 A` Project**
## Hybrid Neural ODEs for Plasma Inductance Dynamics

<center>

| | |
| ---: | :--- |
| **Authors** | [Chris Billingham](https://github.com/caderache2014) |
| | [Phil Prior](mailto:pjr32@uw.edu) |
| | [Tino Wells](https://github.com/tinowells-uwml) |

</center>

### Overview

This project focuses on predicting plasma disruptions in Tokamak fusion reactors, which are sudden losses of confinement that threaten the economic viability of fusion power. Safe termination of a plasma discharge (a "soft landing") requires real-time control of two coupled quantities: (1) $I_p$ plasma current; and (2) $L_i$ internal inductance. 

Building on the work of [Romero (2010)](https://iopscience.iop.org/article/10.1088/0029-5515/50/11/115002) and [Wang et al. (2023)](https://arxiv.org/pdf/2310.20079), this research reproduces a hybrid neural ODE model across three distinct software frameworks to discover interpretable, physically meaningful ordinary differential equations (ODEs) for $V$ dynamics.

[![arXiv:2310.20079](https://img.shields.io/badge/-arXiv%3A2310.20079-white?style=for-the-badge&logo=arxiv&logoColor=white&labelColor=B31B1B)]([https://arxiv.org/abs/2310.20079](https://arxiv.org/abs/2310.20079))
[![DOI: 10.1088/0029-5515/50/11/115002](https://img.shields.io/badge/-10.1088%2F0029--5515%2F50%2F11%2F115002-white?style=for-the-badge&logo=doi&logoColor=white&labelColor=F74E1E)](https://doi.org/10.1088/0029-5515/50/11/115002)


[![GitHub Repo](https://img.shields.io/badge/-hybrid--neural--ode--project-white?style=for-the-badge&logo=github&logoColor=white&labelColor=181717)](https://github.com/caderache2014/ENGR521A.git)
![SciML](https://img.shields.io/badge/-SciML-white?style=for-the-badge&logo=julia&logoColor=white&labelColor=5D87BF)
![PyTorch](https://img.shields.io/badge/-PyTorch-white?style=for-the-badge&logo=pytorch&logoColor=white&labelColor=EE4C2C)
![JAX](https://img.shields.io/badge/-JAX-white?style=for-the-badge&logo=google&logoColor=white&labelColor=4285F4)
![SINDy](https://img.shields.io/badge/-SINDy-white?style=for-the-badge&logo=scikitlearn&logoColor=white&labelColor=F7931E)
![PySR](https://img.shields.io/badge/-PySR-white?style=for-the-badge&logo=dataiku&logoColor=white&labelColor=006699)
![License](https://img.shields.io/badge/-Apache_2.0-white?style=for-the-badge&logo=apache&logoColor=white&labelColor=D22128)


### Repository Structure

```text
ENGR521A/
  ├── data/                           # Immutable inputs
  │   └── romero_shots_489.zip        # C-Mod experimental data
  ├── tokode/                         # Core Python package
  │   ├── __init__.py                 
  │   ├── config.py                   # Configuration and data structures
  │   ├── data.py                     # Data loading and interpolation
  │   ├── models.py                   # PyTorch neural networks and Enum factory
  │   ├── trainer.py                  # Loss functions and the training loop
  │   └── viz.py                      # Matplotlib plotting functions
  ├── training/                       # Execution scripts
  │   ├── train_hybrid.py             # Trains only RomeroNNV
  │   ├── train_physics.py            # Trains only RomeroModel
  │   ├── train_pure_ai.py            # Trains only MlpODE
  │   └── run_all_experiments.py      # End-to-end comparative script
  ├── results/                        # Generated dynamic outputs
  │   ├── figures/                    # Saved matplotlib PNGs/PDFs
  │   ├── logs/                       # Saved .log files from training runs
  │   └── checkpoints/                # Saved PyTorch .pt model weights
  ├── notebooks/                      # Tutorials and interactive exploration
  │   ├── 01_getting_started.ipynb    # Walkthrough of loading data and a basic run
  │   └── 02_model_comparison.ipynb   # Interactive breakdown of the 3 models
  ├── tests/                          # Unit testing suite (pytest)
  │   ├── __init__.py
  │   ├── test_data.py
  │   ├── test_models.py
  │   └── test_trainer.py
  ├── .gitignore                      # Standard ignores for data and model artifacts
  ├── LICENSE                         # Apache 2.0
  ├── README.md                       # Project overview and success metrics
  └── requirements.txt                # Python dependencies
```

---

## Getting Started

### Cloning the Repository

To clone the repository, run the following command in your terminal:

```bash
git clone https://github.com/caderache2014/ENGR521A.git
cd ENGR521A/torch-tokode
```

### Environment Setup

It is highly recommended to use a virtual environment. Install the required dependencies using `pip`:

```bash
python -m venv .venv && source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
pip install -e .
```

### Data Access

This project utilizes the `romero_shots_489` dataset derived from the Alcator C-Mod tokamak reactor at MIT. Ensure the raw data CSV is placed in `data/` before running the training pipelines or notebooks.

### Examples

See the example notebook under [`torch-tokode/examples/train-tokeode-models.ipynb`](./torch-tokode/examples/train-tokeode-models.ipynb) to see an example run.

---

## Success Metrics
* **Reproduction Accuracy**: Reproduce Wang et al. Table 1 with 10% relative error for $L_i$ and $I_p$ predictions on held-out C-Mod shots.
* **SR Performance**: A recovered SR is considered successful if it outperforms the Romero baseline on held-out shots by a statistically meaningful margin.
* **Methodological Agreement**: Agreement between SR and SINDy on the functional form of $\dot{V}$ constitutes a strong positive result.

---

## License

&copy; 2026 Christopher Billingham, Phil Prior, Tino Wells. 

Licensed under the Apache License, Version 2.0 (the [`LICENSE`](./LICENSE)); you may not use this file except in compliance with the License.

You may obtain a copy of the License at:

> <a href="[https://www.apache.org/licenses/LICENSE-2.0](https://www.apache.org/licenses/LICENSE-2.0)"><code>[https://www.apache.org/licenses/LICENSE-2.0](https://www.apache.org/licenses/LICENSE-2.0)</code></a>

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

---

<!-- ## Resources

[![Shared gDrive](https://img.shields.io/badge/-Shared%20gDrive-ECECEC?style=for-the-badge&logo=googledrive&logoColor=white&labelColor=4285F4)]([https://drive.google.com/drive/folders/1v2Y4LL8yl-kPPcwJyEA1VIjGoEWZRw7o](https://drive.google.com/drive/folders/1v2Y4LL8yl-kPPcwJyEA1VIjGoEWZRw7o)) -->