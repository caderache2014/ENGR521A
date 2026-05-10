# Week 7 Update — May 10, 2026

**Author**: Chris Billingham
**Branch**: `chris-update-week7`
**Period covered**: May 4 – May 10, 2026

This update summarizes the work I've done since the May 4 sync. Short version: I built a quantitative diagnostic that probes whether the trained NeuralODEs actually learn the V channel in a physically meaningful way (they do not), then ran a v70 experiment adding V to the loss to test whether that changes the picture (it does not, in any meaningful sense). Together, these results suggest that **symbolic regression on the trained models' V output is not a viable path forward** for our project.

## What I worked on this week

Three main threads, in order:

1. **Built the V_dot correlation diagnostic** to test whether RomeroNN and MlpODE learn V dynamics that reflect the underlying physics, not just whatever V values are needed to make Li and Ip MAPE numbers look good. The diagnostic compares each model's NN-predicted dV/dt against finite-differenced dV/dt computed from the data via cubic-spline derivatives. R², Pearson r, and Spearman ρ are computed over the test set's interior timesteps with middle-99% clipping.
2. **Ran RomeroNN_v70 and (in progress) MlpODE_v70**, which differ from v69 only in adding V to the loss function. The loss adds a Huber term on the variance-normalized V residual with weight λ_V = 0.01 — a pragmatic default, not a tuned value (no time for a hyperparameter sweep).
3. **Produced a publication-quality figure** for the May 8 visualization assignment, showing data + Romero baseline + RomeroNN_v69 + MlpODE_v69 trajectories on shot 1050218005 (panels a–c) plus the V_dot correlation scatter (panel d). See `plots_results/figure_grid_preview_v69_v19.png`.

## Findings

### Cross-implementation MAPE check (Li, Ip)

Applying Phil's endpoint MAPE method (group_size=10, stride=10, error at episode endpoint only) to my Julia models, alongside my own full-shot MAPE, gives the following test-set numbers:

| Metric | Phil's JAX | RomeroNN_v69 (Julia) | MlpODE_v69 (Julia) |
|---|---|---|---|
| Endpoint Li MAPE | 4.02% ± 3.15% | 3.96% ± 5.03% | 4.02% ± 3.44% |
| Endpoint Ip MAPE | 2.42% ± 1.33% | 4.78% ± 6.11% | 2.67% ± 2.59% |

Two takeaways:

- **Endpoint Li MAPE matches Phil's almost exactly** for both Julia models — applying his MAPE method aligned the Li numbers across implementations, confirming the cross-team Li reproduction.
- **Endpoint Ip MAPE matches Phil's closely for MlpODE** (2.67% vs 2.42%) but RomeroNN runs noticeably higher (4.78%). This is an architecture difference, not a methodology difference — the pure neural ODE tracks Ip better than the physics-informed hybrid in our Julia setup.

Phil — credit where due: the endpoint MAPE methodology was the tool that made the Julia/JAX comparison apples-to-apples on Li. Without that, the cross-team agreement we see now would have been hidden behind metric differences.

### V channel diagnostic (the main finding)

The V_dot correlation diagnostic, run on test-set interior points (n=1980 after middle-99% clipping at s=1e-3·N spline smoothing):

| Model | R² | Pearson r | Spearman ρ |
|---|---|---|---|
| Pure Romero physics | 0.0002 | +0.015 | +0.21 |
| RomeroNN_v69 | 0.0066 | -0.081 | -0.20 |
| MlpODE_v69 | 0.0075 | +0.087 | +0.25 |

**All three are essentially zero.** Neither architecture learns dV/dt in a physically meaningful sense. This isn't surprising on reflection: V appears in our state vector but never enters Wang's loss function, only Li and Ip do. The NN is free to push V wherever serves the Li/Ip loss minimization, with no constraint that the V trajectory match the physical V signal.

This is consequential for the SR plan: SR applied to a network whose V-output is uncorrelated with physical dV/dt would extract a non-physical expression. The diagnostic is therefore a necessary gating analysis before any equation-discovery step.

### v70 experiment: V in the loss

To test whether the V failure is *because* V wasn't supervised, I added V to the loss with variance-normalized residuals and λ_V = 0.01:

| Metric | RomeroNN_v69 | RomeroNN_v70 |
|---|---|---|
| V_dot test R² | 0.0066 | 0.0141 |
| V_dot Pearson r | -0.081 | -0.119 |
| Full-shot Li MAPE | 4.62% ± 2.75% | 4.03% ± 3.08% |
| Full-shot Ip MAPE | 5.17% ± 1.99% | 6.28% ± 2.42% |

Reading this honestly:

- **V_dot R² roughly doubled, but both numbers are essentially zero.** Going from "0.7% of variance explained" to "1.4% of variance explained" is not a scientifically meaningful improvement.
- **Pearson r got *more* negative**, not less. A model learning V_dot would show positive correlation with FD V_dot. v70's V output is more strongly anti-correlated with the truth than v69's was.
- **Ip tracking degraded by ~20% relative** as the cost of V supervision. The Huber-weighted gradient pulled the optimizer away from clean Ip fits to accommodate V, without actually getting V right.

We can't fully separate "λ_V = 0.01 was too small" from "V is genuinely unlearnable from this signal" — that would require a hyperparameter sweep we don't have time for. But the direction (or lack of it) is consistent with the underlying problem: the FD V_dot signal is so noisy that even pure Romero physics has R² = 0.0002 against it. There may be no NN configuration that can extract meaningful V dynamics from this data.

### MlpODE_v70: still running

MlpODE_v70 launched today around noon and should finish by tomorrow evening. I'll commit its outputs (BSON checkpoint, derivatives CSV, predictions, MAPE summaries) in a follow-up commit. I expect a similar story to RomeroNN_v70: marginal R² improvement that doesn't change the SR conclusion. But running it for completeness across both architectures is worth the compute time.

## Status, decisions ahead, and the paper

I emailed Dr. Hickner Friday with the v69 diagnostic finding and v70 setup, asking whether to wait for v70 results before deciding on a SINDy pivot. As of this writing she hasn't replied — I expect a response Monday or Tuesday.

My honest read: **pivoting to SINDy from scratch this far into the term is risky** when:

- Our original backup goal (reproduce Wang's results in Julia, with diagnostic analysis) is already largely satisfied.
- The same V_dot noise that defeated the NN would also be the input signal SINDy regresses against. The bottleneck may be the data, not the method.
- The final paper is due in 4 weeks, and writing/figures will eat more time than we anticipate.

The cleaner story may be: "We reproduced Wang's NeuralODE in Julia (and across team in JAX/PyTorch), built a diagnostic that Wang's paper does not report, and showed the V channel is not learned in a physically meaningful way regardless of architecture or whether V is in the loss. SR on the trained models is therefore not viable for V; SINDy on the data inherits the noise problem and is left as future work." That's a finishable, defensible contribution.

But I'd like to hear what you both think, especially after Hickner's reply lands. Let's discuss at the next sync.

## What's in this commit

**Branch**: `chris-update-week7`

**`Julia_SciML/src/`** — latest scripts:
- `LIV_Cmod_RomeroNN_v69.jl`, `LIV_Cmod_MlpODE_v69.jl` — v69 versions (BSON serialization, Phil-style endpoint MAPE, derivatives CSV, all with `safe_spline` fallback)
- `LIV_Cmod_RomeroNN_v70.jl`, `LIV_Cmod_MlpODE_v70.jl` — v70 versions (V added to loss, variance-normalized, λ_V = 0.01)
- `vdot_diagnostic_RomeroNN_v69.jl`, `vdot_diagnostic_MlpODE_v69.jl`, `vdot_diagnostic_RomeroNN_v70.jl` — V_dot correlation diagnostics
- `recover_romeroNN_v70.jl` — post-crash recovery script (RomeroNN_v70 hit a Dierckx spline error during derivatives extraction; this rebuilds the missing outputs from the saved BSON without retraining)
- `preview_figure_grid_v19.jl` — produces the May 8 visualization-assignment figure
- `phils_jax_romeroNN.py` — Phil's JAX reference, kept here for cross-team comparison

**`Julia_SciML/`** root — environment + checkpoints:
- `Project.toml`, `Manifest.toml` — exact Julia package versions used
- `romeroNN_checkpoint_v69.bson`, `romeroNN_checkpoint_v70.bson`, `mlpODE_checkpoint_v69.bson` — trained model parameters

**`Julia_SciML/plots_results/`** — outputs and the publication figure:
- `figure_grid_preview_v69_v19.png` — the May 8 visualization-assignment figure (data + Romero + RomeroNN_v69 + MlpODE_v69 on Li/Ip/V trajectories, plus V_dot scatter)
- `preview_v70_li.png`, `preview_v70_ip.png`, `preview_v70_v.png` — RomeroNN_v70 single-shot trajectory previews
- `predictions_*_v69.csv`, `predictions_*_v70.csv`, `*_predictions_hard_shot_*.csv` — single-shot trajectories (data, Romero, NN) used to build trajectory plots
- `fullshot_mape_TEST_*.csv`, `endpoint_mape_TEST_*.csv` — per-shot/per-episode MAPE values for paired analysis
- `loss_history_*.csv`, `val_loss_history_*.csv` — training and validation loss curves
- `vdot_diagnostic_summary_*.csv` — diagnostic summary statistics across (split × smoothing × clipping × point_filter) configurations

### Provenance notes

- `predictions_RomeroNN_v69.csv` and `RomeroNN_predictions_hard_shot_v69.csv` were sourced from the v68 run (May 3) since RomeroNN's v68 → v69 changes were post-training analysis additions only (BSON serialization, additional MAPE method, derivatives extraction). The trained model is functionally equivalent between v68 and v69.
- `loss_history_RomeroNN_v70.csv` and `val_loss_history_RomeroNN_v70.csv` are reconstructed from the terminal log at every-50-iter granularity, since the v70 script crashed during derivatives extraction before writing the in-memory full-resolution loss histories. v68 and v69 history files are at every-iter granularity. When plotting v69 vs v70 comparisons, down-sample v69 to every-50-iters to match.

### Things gitignored

`.gitignore` updated to exclude:
- Per-point and per-shot V_dot diagnostic CSVs (regenerable; can be ~5 MB each)
- Full derivatives CSVs (regenerable from BSON + scripts)
- Terminal logs and Julia GR backend temp files

Anyone needing the full-detail diagnostic output can regenerate it by running the diagnostic scripts on the committed BSONs.