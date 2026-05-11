# Week 7 Update — May 10, 2026

**Author**: Chris Billingham
**Branch**: `chris-update-week7`
**Period covered**: May 4 – May 10, 2026

This update summarizes the work I've done since the May 4 sync. Short version: I built a quantitative diagnostic that probes whether the trained NeuralODEs actually learn the V channel in a physically meaningful way (they do not). I then ran v70 experiments on both architectures, adding V to the loss to test whether that changes the picture (it does not, across four independent training configurations spanning RomeroNN and MlpODE × v69 and v70). One genuinely positive surprise: MlpODE_v70's Li and Ip MAPE numbers are the best Wang reproduction across the team — better than my v69 numbers, better than Phil's published JAX results — but this happened without learning V physics. Together, these results suggest that **symbolic regression on the trained models' V output is not a viable path forward** for our project, while also producing an unexpectedly strong reproduction of Wang's published Li and Ip dynamics.


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


### Update — MlpODE_v70 completed (May 10 evening)

### MlpODE_v70: completed

MlpODE_v70 completed Sunday May 10 evening (about 32 hours of compute). All output files saved cleanly without the spline-failure crash that hit RomeroNN_v70 — the `safe_spline` fallback worked as designed. Outputs are in this follow-up commit (see "What's in this commit" below). See the "Update — MlpODE_v70 completed" subsection above for full results — short version is that running it for completeness turned out to be more informative than expected.

MlpODE_v70 finished training shortly after the initial draft of this update was committed. The results are surprising in opposite directions across the two metrics that matter:

**MAPE improved dramatically.** MlpODE_v70 produced the best Li and Ip numbers I've seen across the project:

| Metric | MlpODE_v69 | MlpODE_v70 | Change |
|---|---|---|---|
| Full-shot Li MAPE (test) | 4.25% ± 2.68% | 2.97% ± 1.91% | −30% relative |
| Full-shot Ip MAPE (test) | 2.78% ± 2.54% | 1.34% ± 0.69% | −52% relative |
| Endpoint Li MAPE (test) | 4.02% ± 3.44% | 3.00% ± 3.47% | −25% relative |
| Endpoint Ip MAPE (test) | 2.67% ± 2.59% | 1.44% ± 1.47% | −46% relative |

For reference, Phil's JAX endpoint Li MAPE was 4.02% and Ip was 2.42%; MlpODE_v70 now beats both. To my knowledge, **MlpODE_v70 is currently the best Wang-architecture reproduction across our team's three implementations.**

**V_dot diagnostic did not improve.** The same model, on the diagnostic that measures whether NN-predicted dV/dt correlates with finite-differenced physical dV/dt:

| Model | Test R² | Pearson r | Per-shot median R² |
|---|---|---|---|
| Pure Romero physics | 0.0002 | +0.015 | — |
| RomeroNN_v69 | 0.0066 | −0.081 | 0.026 |
| MlpODE_v69 | 0.0075 | +0.087 | 0.068 |
| RomeroNN_v70 | 0.0141 | −0.119 | 0.018 |
| MlpODE_v70 | 0.0062 | −0.079 | 0.080 |

MlpODE_v70's aggregate test R² (0.0062) is actually slightly worse than MlpODE_v69's (0.0075), and the Pearson correlation flipped from weakly positive (+0.087) to weakly negative (−0.079). The per-shot median is the best of any model at 0.080, but the IQR [0.022, 0.196] shows even typical shots have R² below 0.20, and only 2 of 74 test shots cross R² > 0.5.

**Interpretation.** Across four independent training configurations (RomeroNN and MlpODE × v69 and v70), the V_dot R² stays in the 0.006 – 0.014 range. **The result is robust to both architecture choice and to whether V is included in the loss function.** SR on the V output is not viable for any of these models.

The MlpODE_v70 MAPE improvement is best understood as the pure-NN architecture exploiting V as a free variable: V supervision constrained where V values could go, but the network used that freedom to find a configuration that made Li and Ip easier to fit, without recovering V's physical structure. The hybrid architecture (RomeroNN_v70) had less flexibility — its physics-derived Li and Ip equations couldn't be helped by V supervision, so adding V to the loss only degraded Ip tracking. The trade-off across architectures is asymmetric.

This complicates the narrative slightly but doesn't change the SR conclusion. We have:

1. A clean negative result on V_dot, robust across architectures and loss configurations.
2. An unexpectedly strong positive result on Li and Ip MAPE from MlpODE_v70.

Both deserve a place in the final paper.

## Status, decisions ahead, and the paper

I emailed Dr. Hickner Friday with the v69 diagnostic finding and v70 setup, asking whether to wait for v70 results before deciding on a SINDy pivot. As of this writing she hasn't replied — I expect a response Monday or Tuesday.

My honest read: **pivoting to SINDy from scratch this far into the term is risky** when:

- Our original backup goal (reproduce Wang's results in Julia, with diagnostic analysis) is already largely satisfied.
- The same V_dot noise that defeated the NN would also be the input signal SINDy regresses against. The bottleneck may be the data, not the method.
- The final paper is due in 4 weeks, and writing/figures will eat more time than we anticipate.

The cleaner story may now be: "We reproduced Wang's NeuralODE in Julia (and across team in JAX/PyTorch), with MlpODE_v70 producing the best Li and Ip MAPE numbers among all known reproductions. We built a V_dot correlation diagnostic that Wang's paper does not report, and showed the V channel is not learned in a physically meaningful way regardless of architecture or whether V is included in the loss — a result robust across four independent training configurations. SR on the trained models is therefore not viable for V; SINDy on the data inherits the same V_dot noise problem and is left as future work." That's a finishable, defensible contribution, with both a positive numerical result (the MAPE win) and a clean negative result (the SR non-viability) to anchor the paper.

But I'd like to hear what you both think, especially after Hickner's reply lands. Let's discuss at the next sync.

## What's in this commit

**Branch**: `chris-update-week7`

**`Julia_SciML/src/`** — latest scripts:
- `LIV_Cmod_RomeroNN_v69.jl`, `LIV_Cmod_MlpODE_v69.jl` — v69 versions (BSON serialization, Phil-style endpoint MAPE, derivatives CSV, all with `safe_spline` fallback)
- `LIV_Cmod_RomeroNN_v70.jl`, `LIV_Cmod_MlpODE_v70.jl` — v70 versions (V added to loss, variance-normalized, λ_V = 0.01)
- `vdot_diagnostic_MlpODE_v70.jl` — V_dot diagnostic adapted for MlpODE_v70 (follow-up commit)

- `vdot_diagnostic_RomeroNN_v69.jl`, `vdot_diagnostic_MlpODE_v69.jl`, `vdot_diagnostic_RomeroNN_v70.jl` — V_dot correlation diagnostics
- `recover_romeroNN_v70.jl` — post-crash recovery script (RomeroNN_v70 hit a Dierckx spline error during derivatives extraction; this rebuilds the missing outputs from the saved BSON without retraining)
- `preview_figure_grid_v19.jl` — produces the May 8 visualization-assignment figure
- `phils_jax_romeroNN.py` — Phil's JAX reference, kept here for cross-team comparison

**`Julia_SciML/`** root — environment + checkpoints:
- `Project.toml`, `Manifest.toml` — exact Julia package versions used
- `romeroNN_checkpoint_v69.bson`, `romeroNN_checkpoint_v70.bson`, `mlpODE_checkpoint_v69.bson` — trained model parameters
- `mlpODE_checkpoint_v70.bson` — added in follow-up commit (May 10 evening)

**`Julia_SciML/plots_results/`** — outputs and the publication figure:
- `figure_grid_preview_v69_v19.png` — the May 8 visualization-assignment figure (data + Romero + RomeroNN_v69 + MlpODE_v69 on Li/Ip/V trajectories, plus V_dot scatter)
- `preview_v70_li.png`, `preview_v70_ip.png`, `preview_v70_v.png` — RomeroNN_v70 single-shot trajectory previews
- `predictions_*_v69.csv`, `predictions_*_v70.csv`, `*_predictions_hard_shot_*.csv` — single-shot trajectories (data, Romero, NN) used to build trajectory plots
- `fullshot_mape_TEST_*.csv`, `endpoint_mape_TEST_*.csv` — per-shot/per-episode MAPE values for paired analysis
- `loss_history_*.csv`, `val_loss_history_*.csv` — training and validation loss curves
- `vdot_diagnostic_summary_*.csv` — diagnostic summary statistics across (split × smoothing × clipping × point_filter) configurations
- `predictions_MlpODE_v70.csv`, `MlpODE_predictions_hard_shot_v70.csv` — MlpODE_v70 trajectories (follow-up commit)
- `fullshot_mape_TEST_MlpODE_v70.csv`, `endpoint_mape_TEST_MlpODE_v70.csv` — MAPE summaries (follow-up commit)
- `loss_history_MlpODE_v70.csv`, `val_loss_history_MlpODE_v70.csv` — training and validation losses (follow-up commit)
- `vdot_diagnostic_summary_MlpODE_v70.csv` — V_dot diagnostic summary (follow-up commit)

### Provenance notes

- `predictions_RomeroNN_v69.csv` and `RomeroNN_predictions_hard_shot_v69.csv` were sourced from the v68 run (May 3) since RomeroNN's v68 → v69 changes were post-training analysis additions only (BSON serialization, additional MAPE method, derivatives extraction). The trained model is functionally equivalent between v68 and v69.
- `loss_history_RomeroNN_v70.csv` and `val_loss_history_RomeroNN_v70.csv` are reconstructed from the terminal log at every-50-iter granularity, since the v70 script crashed during derivatives extraction before writing the in-memory full-resolution loss histories. v68 and v69 history files are at every-iter granularity. When plotting v69 vs v70 comparisons, down-sample v69 to every-50-iters to match.

### Things gitignored

`.gitignore` updated to exclude:
- Per-point and per-shot V_dot diagnostic CSVs (regenerable; can be ~5 MB each)
- Full derivatives CSVs (regenerable from BSON + scripts)
- Terminal logs and Julia GR backend temp files

Anyone needing the full-detail diagnostic output can regenerate it by running the diagnostic scripts on the committed BSONs.