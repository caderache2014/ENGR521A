# Reproduction of Wang et al. (2023) RomeroNNV on Alcator C-Mod data

ENGR521A Spring 2026 — Christopher Billingham, Tino Wells, Phil Prior

## 1. Project goal

Reproduce the hybrid neural ODE plasma current and internal inductance prediction model from Wang, Garnier, and Rea (2023), in three independent implementations:

- **Julia (SciML)** — Christopher Billingham
- **PyTorch** — Tino Wells
- **JAX (Equinox)** — Phil Prior

Wang's RomeroNNV model is a 3-state ODE for `(li, Ip, V)` over plasma discharges. The `dli/dt` and `dIp/dt` channels are governed by Romero (2010) physics; the `dV/dt` channel is replaced by a neural network. Training data is 489 Alcator C-Mod shots provided by the course.

Wang's published TEST MAPE values (Table 1):
- RomeroNNV: Li 3.50% ± 3.70%, Ip 1.97% ± 1.90%
- MlpODE (pure NN baseline): Li 5.99% ± 7.23%, Ip 8.10% ± 9.69%

## 2. Cross-framework results

| Model / source | TEST Li MAPE | TEST Ip MAPE | Notes |
|---|---|---|---|
| Wang RomeroNNV (paper) | 3.50% ± 3.70% | 1.97% ± 1.90% | Reference target |
| Wang MlpODE (paper) | 5.99% ± 7.23% | 8.10% ± 9.69% | Reference target |
| Julia RomeroNN v68 | 4.62% ± 2.75% | 5.17% ± 1.99% | 7500 AdamW + 500 L-BFGS, full-shot MAPE |
| Julia MlpODE v68 | 4.25% ± 2.68% | 2.78% ± 2.54% | 7500 AdamW + 500 L-BFGS, full-shot MAPE |
| PyTorch RomeroNNV (Tino) | 7.20% | -2.75%* | 256 epochs, 64 train shots; signed denominator |
| JAX RomeroNNV (Phil) | 4.02% ± 3.15% | 2.42% ± 1.33% | 3000 epochs, per-segment endpoint MAPE |

*Tino's reported Ip MAPE is signed (-2.75%) because his denominator uses signed Ip. The true magnitude is ~2.75%; the sign reflects the convention.

## 3. Methodological differences across implementations

The three implementations share Wang's overall approach but differ in non-trivial details. These differences affect comparability of the headline MAPE numbers.

**MAPE definition.**
- Julia (Christopher): MAPE computed across **all 28 timesteps** of each held-out shot, averaged per shot, then aggregated across the 74-shot test split.
- JAX (Phil): MAPE computed at the **final timestep of each 10-step episode** only, averaged across 162 test episodes.
- PyTorch (Tino): Per-shot trapezoidal-integrated relative error over each shot's full trajectory.

These are not equivalent metrics. End-of-segment MAPE on 10-step segments is a much easier evaluation task than full-trajectory MAPE on 28-step shots: errors have less time to compound. Phil's numbers being closer to Wang's may partly reflect that Wang also computes MAPE per-segment rather than per-shot. **This needs clarification from Wang**; the answer affects how we interpret the comparison.

**Train/test split.**
- Julia: shot-level random split (seed=42), 342 train / 73 val / 74 test shots. Each shot is wholly in one split.
- JAX: episode-level sequential split (no shuffling) of a flat list of ~1079 segments. Because all segments from a given shot are contiguous in the list, splits land near shot boundaries — but a few shots have segments straddling the train/val or val/test boundary.
- PyTorch: shot-level random split, 64 train, balance for val (no test split in current code).

**Trainable physics parameters.**
- Julia: κ=0.98 and τ=1.25 (Romero 2010 empirical values) used as fixed constants. Only the NN parameters and `liip_norm`-related quantities are trainable.
- PyTorch: κ and τ are `nn.Parameter` (learnable), initialized to Romero's values.
- JAX: κ and τ not used directly; instead Phil introduces two trainable scaling factors `alpha_li` and `alpha_ip` (initialized to 1.0) that multiply the Romero physics terms. Final values converged to alpha_ip ≈ 1.92, alpha_li ≈ 0.49.

The interpretation of α scaling is open: it acts like a learnable correction on the physics, similar in spirit to learnable κ/τ but mathematically distinct. We don't know which of these (fixed κ/τ, learnable κ/τ, learnable α) corresponds to Wang's actual setup. Wang's JAX snippet has `params: jnp.ndarray = jnp.array([0.98, 1.25])` with `eqx.is_inexact_array` filter, which suggests these are trainable in his code.

**Multi-shooting.**
- Julia: multi-shoot with segment_length=10, stride=9 (overlapping by 1).
- JAX: multi-shoot with group_size=10, stride=10 (non-overlapping). Continuity weight = 0.
- PyTorch: no multi-shooting; full-shot integration.

**Neural network architecture.**
- Julia: 4 → 32 (softplus) → 32 (softplus) → 1, near-zero initialization.
- JAX: same 4 → 32 → 32 → 1 with softplus. Standard initialization.
- PyTorch: 4 → 32 (tanh) → 1, single hidden layer, unbounded output. Simpler than Wang's stated architecture.

**Solver and integration.**
- Julia: Tsit5 adaptive solver via SciML.
- JAX: RK4 with Hermite interpolation, n_substeps=2.
- PyTorch: fixed-step RK4, dt=0.01.

**Additional input features.**
- JAX: includes `Vind_slope` (backward finite difference of Vind) as a fourth feature in controls, used during interpolation. Neither Julia nor PyTorch include this.

## 4. Pending questions for Wang

The methodological divergence above motivates direct outreach to the paper's first author. Three questions ranked by importance:

1. **MAPE methodology.** How are the Table 1 MAPE values computed? Per-shot full-trajectory error, per-segment final-timestep error, or some other definition? On test or validation data? At what training epoch (best validation, or final)?

2. **Trainable physics parameters.** In your JAX code snippet, `RomeroModel.params: jnp.ndarray = jnp.array([0.98, 1.25])` — are κ and τ trainable parameters initialized at Romero's values, or fixed constants? Does the model also include scaling factors (like α_li, α_ip) on the Romero terms?

3. **Normalization.** What is `MachineConstants.for_cmod().romero_norm()` numerically? Is it a fixed constant per machine, or computed from the data?

The team will discuss these questions and finalize the email together at the weekly meeting.

## 5. V channel concerns (across all implementations)

Independent diagnostic comparisons of NN-predicted dV/dt vs finite-differenced dV/dt from data show that **none of the three implementations have learned meaningful V dynamics**:

- Julia: on the standard hard-shot diagnostic, NN-predicted V ranges [-0.10, +0.09] while data V ranges [-2.34, +0.21]. The NN output is approximately zero compared to data variation.
- JAX (Phil's plot): NN V_dot oscillates between -33 and +49 while finite-difference V_dot is in the range [-7, +10]. The NN learns noise on the wrong scale.
- PyTorch: not yet diagnosed but expected to be similar.

This is a structural issue with the training objective. Wang's loss function uses Huber loss on relative errors of `li` and `Ip` only — V is not in the loss. The NN's V output therefore acts as a free variable that the optimizer adjusts to minimize Li/Ip error rather than to match V dynamics. Whatever V values produce the best Li and Ip fits are accepted, even if those V values are unphysical.

This is not a bug in any of our implementations. It's a feature of Wang's published methodology that may need to be raised with the author. It also bears directly on the next phase of the project.

## 6. Looking ahead: equation discovery on the trained NN

The course project's stretch goal is to apply equation discovery — symbolic regression (SR) and/or sparse identification of nonlinear dynamics (SINDy) — to recover an analytic form for the learned `dV/dt` function. This was framed in our proposal as "use the trained NN to generate clean trajectory data, then fit a sparse symbolic expression."

Given the V channel concerns, the practicality of this goal needs reconsideration. Three scenarios are worth thinking through:

**Scenario A: SR on trained RomeroNNV V channel.**
Probe the trained NN at all measured `(Li, Ip, V, Vind)` points to build an (X, y) dataset where y = NN(X). Run SR (e.g. SymbolicRegression.jl) to find a closed-form expression that approximates this function. The risk: if y is mostly near-zero or noise (as the V_dot diagnostics suggest), SR will return either "approximately zero" or a complex expression that overfits noise. Either outcome is unsatisfying as a project finding, though "we attempted SR and discovered the NN learned essentially zero correction" is itself a defensible result.

**Scenario B: SR on trained MlpODE.**
The MlpODE's NN predicts all three derivatives — `dLi/dt`, `dIp/dt`, `dV/dt` — as a function of `(Li, Ip, V, Vind)`. Because Li and Ip are in the loss, those output channels are forced to be meaningful. SR on Li and Ip outputs would test whether MlpODE rediscovered Romero's physics from data, which is a positive scientific finding regardless of how it turns out. SR on the V output is likely as problematic as Scenario A.

**Scenario C: SINDy instead of SR.**
SINDy fits a sparse linear combination of candidate basis functions to *time series* derivatives. It requires regularly-sampled trajectories, which would require generating synthetic data from the trained NN. Trade-off vs SR:
- *SINDy strength*: imposes sparsity, so noisy NN output yields a result that explicitly says "few or no active terms" — easier to interpret.
- *SR strength*: more flexible, can fit any expression form.
- Both fail similarly when the underlying NN output is unreliable, but SINDy's failure mode (sparse zero) is more interpretable than SR's (complex spurious expression).

**Decision pending diagnostic.** The Julia diagnostic script (`LIV_Cmod_RomeroNN_SR_diagnostic_v1.jl`) will compare NN-predicted dV/dt to finite-differenced dV/dt across all 489 shots. The result will determine which scenario to pursue:
- Strong correlation (R² > ~0.5) → proceed with Scenario A (SR on RomeroNNV V channel).
- Weak correlation → pivot to Scenario B (SR on MlpODE) or Scenario C (SINDy on Li/Ip channels of MlpODE, treating V as auxiliary).

Phil's V_dot plot (NN output uncorrelated with finite-difference) is consistent with what to expect from a Julia diagnostic. If the Julia result is similar, the team should jointly decide on the pivot before spending time on SR on the RomeroNNV V channel.

## 7. Status and next steps (week of 4 May 2026)

**Completed:**
- Julia RomeroNN v68: 7500 + 500 L-BFGS, loss-history saved, TEST MAPE 4.62% / 5.17%.
- JAX RomeroNNV: working model with reported MAPE 4.02% / 2.42% on Phil's per-segment metric.
- PyTorch RomeroNNV: working model with reported MAPE 7.20% / 2.75%.

**In progress:**
- Julia MlpODE v68: cloned and queued for overnight 7500 + 500 L-BFGS run; loss-history collection added.
- Julia diagnostic for V channel: script not yet written. Will require BSON serialization of trained θ from RomeroNN v69 (clone of v68 plus checkpoint save).

**Pending team discussion (4 May meeting):**
- Reconcile MAPE definitions: agree on a single computation and re-evaluate all three implementations using it.
- Reconcile trainable parameters: agree on whether κ and τ should be trainable, fixed, or replaced by α scaling.
- Agree on questions for Wang and finalize email.
- Agree on equation discovery direction (SR vs SINDy, RomeroNNV vs MlpODE).
- Document team-wide methodology in a shared written reference (this README or a follow-up).

## X. MlpODE outperforms RomeroNN — an unexpected result

The Julia MlpODE v68 results are notable on two fronts:

**1. Julia MlpODE substantially beats Wang's reported MlpODE.**
Wang's MlpODE: Li 5.99% ± 7.23%, Ip 8.10% ± 9.69%.
Julia MlpODE v68: Li 4.25% ± 2.68%, Ip 2.78% ± 2.54%.

This is a 1.7pp improvement on Li and a 5.3pp improvement on Ip relative to Wang's published numbers. Possible explanations include differences in training duration (we ran 7500 AdamW + 500 L-BFGS), use of multi-shooting, or hyperparameter choices. A direct comparison of training setups would clarify this — see questions for Wang in §4.

**2. Julia MlpODE outperforms Julia RomeroNN on Ip.**
Within our Julia implementations, MlpODE achieves Ip MAPE of 2.78% versus RomeroNN's 5.17% — a 2.4pp gap favoring the pure-NN model over the physics-NN hybrid. This is the *opposite* of the ordering in Wang's Table 1, where RomeroNNV beats MlpODE on Ip (1.97% vs 8.10%).

A working hypothesis: our RomeroNN uses fixed Romero (2010) coefficients (κ=0.98, τ=1.25) that may not be optimal for C-Mod data. MlpODE has no such constraint and is free to learn whatever Li/Ip dynamics best fit the data. If the fixed physics coefficients are slightly wrong for this dataset, RomeroNN inherits a small handicap that MlpODE doesn't.

This hypothesis is testable. Tino's PyTorch implementation treats κ and τ as learnable; Phil's JAX implementation uses learnable α scaling factors instead of κ/τ. If the fixed-coefficient hypothesis is right, the implementations with trainable physics parameters should not show the same MlpODE-beats-RomeroNN ordering. Comparison across implementations would be informative once methodologies are aligned.

**Implications for the project's equation-discovery phase.**
This finding strengthens the case for performing symbolic regression on the trained MlpODE rather than on the trained RomeroNN V channel. MlpODE's NN learns Li and Ip dynamics directly under loss supervision, and our results show those learned dynamics are accurate. SR on those channels could either rediscover Romero-like physics (a positive validation of Romero's framework) or reveal that something different from Romero better fits this data (a more interesting finding). See §6 for the equation-discovery decision tree.
