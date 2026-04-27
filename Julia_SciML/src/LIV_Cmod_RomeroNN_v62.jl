# ============================================================================
# RomeroNN_v62.jl — Hybrid neural ODE for plasma current and inductance
# dynamics in the Alcator C-Mod tokamak.
#
# Physics-informed neural ODE following Wang, Garnier & Rea (2023).
# The first two equations of the Romero (2010) lumped-parameter model are
# kept exact. The third equation (V-dot) is augmented with a residual
# neural network. Trained on 489 plasma discharges from Alcator C-Mod.
# ============================================================================

# --- Numerical / ML stack -----------------------------------------------------
using Lux                       # neural network layers
using Flux: destructure          # flatten/restore NN parameters
using ComponentArrays
using Optimisers                 # AdamW, etc.
using Optimization               # higher-level optimization wrapper
using OptimizationOptimisers     # Optimization <-> Optimisers bridge
using OptimizationOptimJL        # L-BFGS adapter (currently unused; reserved)
using OptimizationPolyalgorithms

# --- ODE / sensitivity stack --------------------------------------------------
using OrdinaryDiffEq             # Tsit5 and other ODE solvers
using DiffEqFlux                 # neural ODE utilities
using SciMLSensitivity           # adjoint methods for ODE gradients

# --- Data, interpolation, plotting --------------------------------------------
using Glob                       # file globbing
using DataFrames, CSV
using Dierckx                    # 1D cubic spline interpolation of Vind
using Plots
using Random

using Statistics       # for mean, std, etc.
using SciMLBase        # for ReturnCode and other SciML types
using Zygote 
using Random: MersenneTwister

# ============================================================================
# DATA LOADING
# ============================================================================
# We load 489 individual CSV files, one per plasma discharge ("shot"), from
# the romero_shots_489 directory. Each file is roughly 30 rows × 11 columns
# of state and control measurements at irregular ~20 ms intervals.
#
# Column meanings (physical quantities):
#   shot         : Alcator C-Mod shot number (e.g., 1050218005)
#   time         : measurement timestamp [s]
#   ip_MA        : plasma current [MA]              — state variable
#   li           : plasma internal inductance [-]   — state variable
#   vc_minus_vb  : V (Romero summary statistic) [V] — state variable
#   Vind         : inductive loop voltage [V]       — control variable
#   kappa, tau   : (legacy) Romero free parameters; not used at runtime
#   dip_dt, dli_dt : (legacy) finite-difference derivatives; not used here
# ----------------------------------------------------------------------------

# NOTE: this path is laptop-specific. To make the script portable, change
# this to a relative path or accept it as an argument.
path_to_csv_dir = "/Users/cmason83/Desktop/sciml_workspaces/wang_model/wang_model_workspace/src/romero_shots_489"

csv_files = glob("*.csv", path_to_csv_dir)             # all 489 shot CSVs
df_list = DataFrame.(CSV.File.(csv_files))             # parse each as DataFrame
df_489_shots = reduce(vcat, df_list)                   # concatenate into one


# ============================================================================
# DIAGNOSTIC SHOT SETUP
#
# A single shot is extracted up front for two reasons:
#   1. Sanity checks on the data loading pipeline (length, monotonicity, etc.)
#   2. End-of-script comparison plots and CSV output (model vs. data on
#      one specific shot, used for visual debugging across script versions).
#
# Training itself does NOT use these single-shot variables. Training builds
# its own per-shot data and per-shot Vind splines (see "TRAINING DATA
# PRECOMPUTE" section below).
# ============================================================================

const T = Float32   # all training parameters/states are Float32

# Shot 1050218005 was chosen as a representative ramp-down. It has 28
# timesteps spanning ~0.5 s and exhibits the late-shot V excursion that
# stresses the V-dot equation.
shot_id   = 1050218005
one_shot  = df_489_shots[df_489_shots.shot .== shot_id, :]

t_shot    = T.(one_shot.time .- one_shot.time[1])   # relative time, starts at 0
li_shot   = T.(one_shot.li)
ip_shot   = T.(one_shot.ip_MA)
V_shot    = T.(one_shot.vc_minus_vb)
Vind_shot = T.(one_shot.Vind)

@assert all(diff(t_shot) .> 0) "time must be strictly increasing"
@show length(t_shot)        # 28 for shot 1050218005
@show t_shot[end]           # ≈ 0.47 s

# Cubic spline of the control variable Vind for THIS SHOT ONLY.
# Used by diagnostic plotting; per-shot training splines are built later.
# `s` controls smoothing strength; ~1e-3 × N gives mild smoothing of noise.
N         = length(t_shot)
smoothing = 1e-3 * N
spl_shot  = Spline1D(t_shot, Vind_shot; k=3, bc="extrapolate", s=smoothing)
Vind_spline32(t) = T(spl_shot(t))   # Float32-typed spline evaluator

# ============================================================================
# PHYSICS CONSTANTS AND ROMERO PARAMETERS
#
# liip_norm: scaling factor used in the Romero ODE denominators. Defined
#            from the maximum |Ip| and |li| in the entire 489-shot dataset
#            so that the denominators of dLi/dt and dIp/dt remain O(1).
# κ_empirical, τ: the two free parameters of Romero's V-dot equation.
#                 Values 0.98 and 1.25 are taken from Wang's hard-coded
#                 initial guess, originally fit by Romero (2010) on JET.
# ============================================================================
max_ip_MA   = maximum(abs.(T.(df_489_shots.ip_MA)))
max_li      = maximum(abs.(T.(df_489_shots.li)))
liip_norm   = T(max_ip_MA * max_li)
κ_empirical = T(0.98)
τ           = T(1.25)

# Initial condition and parameter NamedTuple for the diagnostic single-shot
# Romero solve. Used in the comparison plots at the bottom of this script.
u₀ = T[li_shot[1], ip_shot[1], V_shot[1]]
params_romero32 = (k = κ_empirical, τ = τ, liip_norm = liip_norm)

@show u₀
@show liip_norm


@show minimum(abs.(li_shot)), minimum(abs.(ip_shot))

# ============================================================================
# NUMERICAL HELPERS
# ============================================================================
# Smooth absolute value: avoids the sharp corner of |x| at zero, which would
# break Zygote's reverse-mode autodiff. EPS_DENOM is added to the Romero
# ODE denominators to keep them well-defined when Li or Ip cross zero.
const EPS_DENOM = T(1e-9)
softabs(x, eps_=T(1e-9)) = sqrt(x*x + eps_*eps_)


# ============================================================================
# PURE-ROMERO ODE
#
# The first two equations come from energy conservation and flux balance
# and are exact (Romero 2010). The third (V-dot) is a first-order
# approximation with two free parameters κ ≈ 0.98, τ ≈ 1.25 (originally
# fit on JET).
#
# Sign convention follows Romero 2010 Eq. (45):
#     dV/dt = -V/τ - (κ/τ)·Vind
# (Note: V_R - V_B = -V_ind here. This sign was wrong in earlier script
#  versions and was corrected at v54; do not flip without re-deriving.)
#
# IMPORTANT: the same three equations are inlined in `dudt_romeroNN_diag`
# below and inside `loss_multishot`'s local `dudt_shot`. If you change one,
# update all three.
# ============================================================================
function romero_ODEfunc!(du::AbstractVector{T}, u::AbstractVector{T}, p, t::T)
    Li, Ip, V = u
    Vind = Vind_spline32(t)
    dli_dt_denom = p.liip_norm * (abs(Ip) + EPS_DENOM)
    dip_dt_denom = p.liip_norm * (abs(Li) + EPS_DENOM)
    du[1] = (-2*Vind - 2*V) / dli_dt_denom
    du[2] = ( 2*Vind +    V) / dip_dt_denom
    du[3] = -V/p.τ - (p.k/p.τ)*Vind
    return nothing
end


# ============================================================================
# DIAGNOSTIC SINGLE-SHOT ODE PROBLEM (TODO: relocate near comparison plots)
#
# Builds an ODEProblem for the chosen diagnostic shot using pure Romero
# physics. The solution `true_romero_data` and the problem `romero_trueode`
# are both used by the bottom-of-script comparison plots that compare data
# vs. pure Romero vs. trained NN on shot 1050218005.
# ============================================================================
tspan32        = (t_shot[1], t_shot[end])
romero_trueode = ODEProblem(ODEFunction(romero_ODEfunc!), u₀, tspan32, params_romero32)

@assert romero_trueode.f isa SciMLBase.ODEFunction{true}  "romero_trueode must be in-place"


# ============================================================================
# SENSITIVITY METHOD AND NON-DIFFERENTIABLE GLOBALS
# ============================================================================
# ZYG: the adjoint method passed to `solve` so Zygote can backpropagate
# through ODE solves during training.
ZYG = InterpolatingAdjoint(autojacvec = ZygoteVJP())

# Tell Zygote not to attempt to differentiate through these:
#   Vind_spline32: fixed interpolant of measured data, no learnable params.
#   (The training loss has its own per-shot splines; same logic applies.)
Zygote.@nograd Vind_spline32

# Pure-Romero solution on the diagnostic shot (used by comparison plots).
true_romero_data = Array(solve(romero_trueode, Tsit5();
    saveat=t_shot, reltol=1e-6, abstol=1e-8, verbose=false))


# ============================================================================
# NEURAL NETWORK FOR THE V-DOT RESIDUAL
# ============================================================================
# Architecture: 4 → 32 → 32 → 1, matching Wang et al. (2023).
# Inputs:  [Li, Ip, V, Vind]
# Output:  scalar correction added to the Romero V-dot equation.
#
# Final layer uses a scaled tanh (output bounded to ±5). Wang's reference
# implementation uses unbounded identity output, but in our experiments
# an unbounded NN output destabilized the ODE solver on ~43% of shots
# (v58). ±5 is large enough not to be the limiting factor in residuals
# while keeping the solver stable (v59 onwards).
Random.seed!(42)
rng1 = Random.default_rng()

romero_NN = Lux.Chain(
    Lux.Dense(4, 32, softplus),
    Lux.Dense(32, 32, softplus),
    Lux.Dense(32, 1, x -> T(5.0) * tanh(x))
)

p_nn0, st_nn = Lux.setup(rng1, romero_NN)
Zygote.@nograd st_nn   # st_nn is empty for stateless Dense layers

# Flatten params for use with Optimization.jl, which expects a flat vector.
θ0, re = destructure(p_nn0)
θ0 = T.(θ0)

# Near-zero init of the final layer so the NN starts producing ~0 residual.
# This makes initial behavior approximately equal to pure Romero, giving
# the optimizer a sensible starting point. Final Dense(32,1) has 32
# weights + 1 bias = 33 trailing parameters in θ0.
θ0[end-32:end] .= T(0.01) .* θ0[end-32:end]


# ============================================================================
# DIAGNOSTIC NN ODE PROBLEM (TODO: relocate near comparison plots)
#
# Used only by the bottom-of-script comparison plot that runs the trained
# NN on the diagnostic shot. NOT used during training — training defines
# its own per-shot dudt closures inside loss_multishot.
# ============================================================================
function dudt_romeroNN_diag(du, u, θ, t)
    Li, Ip, V = u
    Vind = Vind_spline32(t)
    du[1] = (-2*Vind - 2*V) / (liip_norm * softabs(Ip))
    du[2] = ( 2*Vind +    V) / (liip_norm * softabs(Li))
    ps = re(θ)
    x = reshape(T[Li, Ip, V, Vind], 4, 1)
    y, _ = Lux.apply(romero_NN, x, ps, st_nn)
    du[3] = -V/τ - (κ_empirical/τ)*Vind + y[1]
end

prob_pred_romeroNN = ODEProblem(dudt_romeroNN_diag, u₀, tspan32, θ0)

# ============================================================================
# TRAINING-TIME GLOBALS (loss history, iteration counter)
# Populated by the callback during training. losses[] is the per-iteration
# training loss; iter is incremented every time the optimizer calls back.
# ============================================================================
losses = Float32[]
iter   = 0

# Note: the Huber loss is defined inline inside loss_multishot below,
# matching Wang's optax.huber_loss(delta=0.1) on relative errors.


# ============================================================================
# DATA SUBSET AND TRAIN/VAL/TEST SPLIT
#
# All 489 shots are used. The split is by SHOT, not by timestep — this
# matches Wang's setup and ensures the model is evaluated on plasma
# discharges it has never seen any portion of.
#
# Split fractions (0.70 / 0.15 / 0.15) and seed (42) match Wang's JAX
# configuration. The seed is fixed so the same shot IDs land in the same
# splits across every run, making versioned comparisons valid.
# ============================================================================
N_shots   = 489
shot_ids  = unique(df_489_shots.shot)[1:N_shots]

split_rng         = MersenneTwister(42)
shot_ids_shuffled = shuffle(split_rng, copy(shot_ids))

n_total = length(shot_ids_shuffled)
n_train = Int(floor(0.70 * n_total))
n_val   = Int(floor(0.15 * n_total))
n_test  = n_total - n_train - n_val   # remainder absorbs rounding

train_shot_ids = shot_ids_shuffled[1 : n_train]
val_shot_ids   = shot_ids_shuffled[n_train+1 : n_train+n_val]
test_shot_ids  = shot_ids_shuffled[n_train+n_val+1 : end]

println("=== Train/Val/Test split (seed=42) ===")
println("Train: $(length(train_shot_ids)) shots")
println("Val:   $(length(val_shot_ids)) shots")
println("Test:  $(length(test_shot_ids)) shots")
println("Total: $n_total shots (original), $(n_train+n_val+n_test) after split")
println("First 3 train shot IDs: $(train_shot_ids[1:3])")
println("First 3 val   shot IDs: $(val_shot_ids[1:3])")
println("First 3 test  shot IDs: $(test_shot_ids[1:3])")


# ============================================================================
# PER-SHOT PRECOMPUTE: VIND SPLINES AND STATE/CONTROL ARRAYS
#
# Both dicts are keyed by shot ID and built once before training. Doing
# the spline construction per-shot up front (rather than inside the loss)
# avoids rebuilding 489 splines on every forward pass.
#
# shot_splines[sid] : cubic spline of Vind for shot `sid`, used inside
#                     the ODE RHS to interpolate the control between
#                     measured timesteps.
# shot_data[sid]    : NamedTuple with measured trajectories needed by the
#                     loss (initial conditions, targets, and V at segment
#                     boundaries for multi-shooting).
# ============================================================================
shot_splines = Dict{Int64, Spline1D}()
for sid in shot_ids
    shot   = df_489_shots[df_489_shots.shot .== sid, :]
    t_s    = T.(shot.time .- shot.time[1])
    Vind_s = T.(shot.Vind)
    shot_splines[sid] = Spline1D(t_s, Vind_s; k=3, bc="extrapolate",
                                  s = Float64(1e-3) * length(t_s))
end

shot_data = Dict{Int64, NamedTuple}()
for sid in shot_ids
    shot = df_489_shots[df_489_shots.shot .== sid, :]
    t_s  = T.(shot.time .- shot.time[1])
    shot_data[sid] = (
        t_s   = t_s,                                          # relative timestamps
        u0_s  = T[shot.li[1], shot.ip_MA[1], shot.vc_minus_vb[1]],  # initial state
        li    = T.(shot.li),
        ip_MA = T.(shot.ip_MA),
        V     = T.(shot.vc_minus_vb),   # measured V; needed at segment boundaries
    )
end


# ============================================================================
# TRAINING LOSS — MULTI-SHOOTING WITH HUBER ON RELATIVE ERROR
#
# For each training shot, the time series is chopped into 10-step segments
# (matching Wang's segment_length=10). Each segment is integrated forward
# from its measured initial state independently, and Huber loss is computed
# on the relative error of Li and Ip predictions vs. measurements within
# that segment. V is not directly supervised; it is only constrained
# indirectly through how it enters the dLi/dt and dIp/dt equations.
#
# Multi-shooting decomposes a long-horizon optimization (28+ steps) into
# many short-horizon problems (10 steps), which dramatically improves
# gradient quality through the ODE solver. Wang reports this is essential
# for stable training of plasma neural ODEs.
#
# Continuity between segments is NOT enforced (matching Wang's "zero
# continuity weight"). Each segment starts from the measured state, so
# discontinuities at boundaries don't accumulate error.
#
# IMPORTANT: the V-dot equation inside `dudt_shot` below must match
# `romero_ODEfunc!` and `dudt_romeroNN_diag` in sign and scaling.
# ============================================================================
function loss_multishot(θ)
    total_loss = 0.0f0
    n_valid_segments = 0
    segment_length = 10   # Wang's group_size

    for sid in train_shot_ids
        d     = shot_data[sid]
        t_s   = d.t_s
        spl_s = shot_splines[sid]
        N     = length(t_s)

        # Stack measured Li, Ip, V into a 3×N matrix so we can pull the
        # initial condition for each segment out as a single column slice.
        u_measured = hcat(d.li, d.ip_MA, d.V)'

        # Segment starts overlap by one timestep at boundaries.
        seg_starts = 1 : segment_length-1 : N-1

        # Hybrid Romero+NN ODE RHS for THIS shot. Captures `spl_s` and `θ`
        # in its closure. NOTE: must stay in sync with `romero_ODEfunc!`
        # and `dudt_romeroNN_diag` for the physics terms.
        function dudt_shot(du, u, θ, t)
            Li, Ip, V = u
            Vind = T(Zygote.ignore(() -> spl_s(t)))
            du[1] = (-2*Vind - 2*V) / (liip_norm * softabs(Ip))
            du[2] = ( 2*Vind +    V) / (liip_norm * softabs(Li))
            ps   = re(θ)
            x    = reshape(T[Li, Ip, V, Vind], 4, 1)
            y, _ = Lux.apply(romero_NN, x, ps, st_nn)
            du[3] = -V/τ - (κ_empirical/τ)*Vind + y[1]
        end

        for i_start in seg_starts
            i_end = min(i_start + segment_length - 1, N)
            i_end - i_start < 2 && continue   # skip degenerate tail segments

            u0_seg = u_measured[:, i_start]   # measured initial state
            t_seg  = t_s[i_start : i_end]

            prob_s = ODEProblem(dudt_shot, u0_seg, (t_seg[1], t_seg[end]), θ)
            sol = solve(prob_s, Tsit5(), p = θ, saveat = t_seg,
                        sensealg = ZYG, reltol = 1f-4, abstol = 1f-6,
                        verbose = false)

            if sol.retcode == ReturnCode.Success
                pred = Array(sol)
                eps_l = T(1e-9)
                li_seg = d.li[i_start : i_end]
                ip_seg = d.ip_MA[i_start : i_end]
                # Relative error, signed denominator (matches Wang's loss).
                r1 = (li_seg .- pred[1, :]) ./ (li_seg .+ sign.(li_seg) .* eps_l)
                r2 = (ip_seg .- pred[2, :]) ./ (ip_seg .+ sign.(ip_seg) .* eps_l)
                # Inline Huber with δ = 0.1 (matches Wang's outlier_threshold).
                hub(x) = x ≤ T(0.1) ? T(0.5) * x^2 / T(0.1) : x - T(0.05)
                total_loss += sum(map(hub, r1) .+ map(hub, r2))
                n_valid_segments += 1
            end
        end
    end

    return n_valid_segments > 0 ? total_loss / n_valid_segments : T(1000.0)
end


# ============================================================================
# VALIDATION LOSS AND BEST-CHECKPOINT TRACKING
#
# During training, the validation loss is evaluated every
# `n_epochs_per_val` iterations on the held-out val_shot_ids. The best
# parameter vector seen so far (lowest val loss) is kept in `best_θ[]`,
# and at the end of training θ_trained is set to this checkpoint rather
# than the final state. This protects against overshoot late in training.
#
# Note on scales: val_loss and the training loss (loss_multishot) use
# different normalizations and are NOT on the same absolute scale.
#   - Training loss is summed over multi-shoot segments, normalized
#     per segment.
#   - Val loss is full-shot integration (no multi-shooting), normalized
#     per shot.
# Compare each to its own baseline; do not compare across.
# ============================================================================
best_val_loss = Ref(Inf32)
best_θ        = Ref(copy(θ0))
val_history   = Tuple{Int, Float32}[]

# val_loss mirrors loss_multishot's RHS but integrates each val shot in
# full (no multi-shooting) and skips Zygote autodiff (pure forward eval,
# no gradients flow through this).
#
# IMPORTANT: dudt_val below must stay in sync with `dudt_shot` in
# loss_multishot, `dudt_romeroNN_diag`, and `romero_ODEfunc!`.
function val_loss(θ)
    total_loss = 0.0f0
    n_valid    = 0
    for sid in val_shot_ids
        d     = shot_data[sid]
        spl_s = shot_splines[sid]

        function dudt_val(du, u, θ, t)
            Li, Ip, V = u
            Vind = T(spl_s(Float64(t)))
            du[1] = (-2*Vind - 2*V) / (liip_norm * softabs(Ip))
            du[2] = ( 2*Vind +    V) / (liip_norm * softabs(Li))
            ps   = re(θ)
            x    = reshape(T[Li, Ip, V, Vind], 4, 1)
            y, _ = Lux.apply(romero_NN, x, ps, st_nn)
            du[3] = -V/τ - (κ_empirical/τ)*Vind + y[1]
        end

        prob_v = ODEProblem(dudt_val, d.u0_s, (d.t_s[1], d.t_s[end]), θ)
        sol    = solve(prob_v, Tsit5(), p = θ, saveat = d.t_s,
                       reltol = 1f-4, abstol = 1f-6, verbose = false)

        if sol.retcode == ReturnCode.Success
            pred = Array(sol)
            eps_l = T(1e-9)
            # NOTE: this uses abs denominators while loss_multishot uses
            # signed denominators. Mathematically equivalent under Huber
            # (symmetric in argument); kept for now to avoid changing
            # existing v55–v61 val-loss numbers. TODO: unify.
            r1 = abs.(d.li    .- pred[1, :]) ./ (abs.(d.li)    .+ eps_l)
            r2 = abs.(d.ip_MA .- pred[2, :]) ./ (abs.(d.ip_MA) .+ eps_l)
            hub(x) = x ≤ T(0.1) ? T(0.5) * x^2 / T(0.1) : x - T(0.05)
            total_loss += sum(map(hub, r1) .+ map(hub, r2)) / length(d.t_s)
            n_valid    += 1
        end
    end
    return n_valid > 0 ? total_loss / n_valid : T(1000.0)
end


# ============================================================================
# OPTIMIZATION.JL CALLBACK
#
# Called once per AdamW iteration. Records the training loss, prints
# every 100 iters, and (every n_epochs_per_val iters) runs val_loss on
# the current parameter vector to update the best checkpoint.
#
# `_get_θ` exists because Optimization.jl's callback API has shifted
# between versions. Older versions pass the raw parameter vector as
# the first argument; newer versions pass a state object with `.u`
# holding the parameter vector. Detecting at runtime keeps the script
# working across the version changes we've seen.
# ============================================================================
_get_θ(p) = hasproperty(p, :u) ? p.u : p

iter             = 0
n_epochs_per_val = 50
callback = function(p, l; doplot = false)
    push!(losses, l)
    global iter
    iter += 1

    if iter % 100 == 0
        println("Iter $iter: train_loss = $l")
    end

    if iter % n_epochs_per_val == 0
        θ_current = _get_θ(p)
        vl = val_loss(θ_current)
        push!(val_history, (iter, vl))
        if vl < best_val_loss[]
            best_val_loss[] = vl
            best_θ[]        = copy(θ_current)
            println("  Iter $iter: val_loss = $vl  ← new best")
        else
            println("  Iter $iter: val_loss = $vl")
        end
    end

    return false
end

# ============================================================================
# OPTIMIZATION PROBLEM SETUP
# Wraps loss_multishot in the form Optimization.jl expects: a function of
# (params, p) where `p` is unused (placeholder for future hyperparameters).
# AutoZygote requests reverse-mode autodiff via Zygote.
# ============================================================================
adtype  = Optimization.AutoZygote()
optf    = Optimization.OptimizationFunction((x, p) -> loss_multishot(x), adtype)
optprob = Optimization.OptimizationProblem(optf, θ0)


# ============================================================================
# ADAMW TRAINING WITH EXPONENTIAL LR DECAY (CHUNKED-SOLVE APPROXIMATION)
#
# Wang's schedule (optax.exponential_decay):
#     η(step) = max(1e-3 * 0.9^(step/50), 2.5e-4)
# Decay updates every 50 steps; floor of 2.5e-4 is reached at step ≈ 658.
#
# Optimization.solve does not accept a schedule callable as the AdamW
# learning rate, and Optimisers.adjust! on the internal optimizer state
# is not exposed via the callback. We instead approximate the schedule
# by calling solve repeatedly in 50-iter chunks, each with a fresh AdamW
# constructed with the LR appropriate to that chunk's starting step.
#
# Tradeoff: the staircase of LR values matches Wang's exactly (he also
# only updates every 50 steps), but the AdamW momentum state resets at
# each chunk boundary, where Wang's optax keeps it continuous. In practice
# the resulting brief instability at each boundary is small (~30 of them
# in a 1500-iter run, each recovering within a few iterations).
# ============================================================================
function wang_lr(step::Integer)
    raw = 1f-3 * 0.9f0 ^ (step / 50f0)
    return max(raw, 2.5f-4)
end

const total_target_iters = 1000   # scouting run; bump to 1500/5000 for full
const chunk_size         = 50

current_θ  = θ0
total_done = 0

while total_done < total_target_iters
    # `global` is required here because `while` is "soft scope" at top
    # level. Without it, Julia treats current_θ and total_done as new
    # locals and the while condition crashes on iteration 2.
    global current_θ, total_done

    chunk = min(chunk_size, total_target_iters - total_done)
    lr    = wang_lr(total_done)
    println("──── Chunk start: iter $(total_done+1), LR = $lr ────")

    optprob_chunk = Optimization.OptimizationProblem(optf, current_θ)
    res_chunk = Optimization.solve(optprob_chunk,
        OptimizationOptimisers.AdamW(lr, (0.9f0, 0.999f0), 1f-4);
        callback = callback,
        maxiters = chunk)

    current_θ   = res_chunk.minimizer
    total_done += chunk
end

# `Optimization.solve` returns an `OptimizationSolution` with a `.minimizer`
# field. The chunked loop above produces a plain vector instead, so we
# wrap it in a one-field NamedTuple to expose the same `.minimizer`
# interface that downstream code may rely on.
res_ms = (; minimizer = current_θ)
println("\n=== AdamW training complete: $total_done iterations done ===")


# ============================================================================
# PHASE 2: L-BFGS REFINEMENT (currently disabled)
#
# Wang's pipeline ends at AdamW. Earlier versions of this script chained
# L-BFGS after AdamW for a few hundred extra iterations as a fine-tuning
# pass. The marginal gain in v53 was negligible (loss 0.17093 → 0.17092),
# so it's been kept off pending evidence it helps. To re-enable, uncomment
# the block below and rerun.
# ============================================================================
#=
println("Starting L-BFGS refinement phase...")
optprob_lbfgs = Optimization.OptimizationProblem(optf, res_ms.minimizer)
res_lbfgs = Optimization.solve(optprob_lbfgs,
    OptimizationOptimJL.LBFGS();
    callback = callback,
    maxiters = 500)
println("L-BFGS final loss: $(losses[end])")
=#


# ============================================================================
# FINAL CHECKPOINT SELECTION
# Use the best-val-loss parameters as the trained model, NOT the final
# AdamW state. This is the protection against late-training overshoot
# discussed in the val-loss section above.
# ============================================================================
println("\n=== Training complete ===")
println("Final train loss:  $(losses[end])")
println("Best val loss:     $(best_val_loss[])  ($(length(val_history)) val evaluations)")
θ_trained = best_θ[]

# ============================================================================
# PER-SPLIT MAPE EVALUATION
#
# Runs both the pure-Romero baseline and the trained NN on every shot in
# a given split (train, val, or test) and reports mean absolute percentage
# error (MAPE) on Li and Ip. The same metric Wang reports in his Table 1.
#
# Returns per-shot error arrays so downstream analysis can compute paired
# statistics (e.g., per-shot NN-vs-Romero comparisons).
#
# IMPORTANT: this function defines TWO inline ODE RHS closures
# (`romero_ODEfunc_f64!` and `dudt_eval`). They must stay in sign-and-
# scaling sync with `romero_ODEfunc!`, `dudt_romeroNN_diag`, `dudt_shot`
# (inside loss_multishot), and `dudt_val` (inside val_loss).
# ============================================================================
function evaluate_split(shot_id_list, split_name::String)
    romero_li_errors = Float32[]
    romero_ip_errors = Float32[]
    nn_li_errors     = Float32[]
    nn_ip_errors     = Float32[]
    n_failed_romero  = 0
    n_failed_nn      = 0

    for sid in shot_id_list
        d     = shot_data[sid]
        spl_s = shot_splines[sid]

        # Pure-Romero baseline solve in Float64. Float32 was numerically
        # unstable for the pure-Romero path on a fraction of shots in v48,
        # producing NaN trajectories and excluding ~90% of shots from MAPE.
        # The NN-augmented path stays in Float32; only this baseline is
        # promoted to Float64.
        function romero_ODEfunc_f64!(du, u, p, t)
            Li, Ip, V = u
            Vind = spl_s(Float64(t))
            eps64 = 1e-9
            dli_denom = p.liip_norm * (abs(Ip) + eps64)
            dip_denom = p.liip_norm * (abs(Li) + eps64)
            du[1] = (-2*Vind - 2*V) / dli_denom
            du[2] = ( 2*Vind +   V) / dip_denom
            du[3] = -V/p.τ - (p.k/p.τ)*Vind
        end

        prob_r64 = ODEProblem(
            romero_ODEfunc_f64!,
            Float64.(d.u0_s),
            (Float64(d.t_s[1]), Float64(d.t_s[end])),
            (k         = Float64(κ_empirical),
             τ         = Float64(τ),
             liip_norm = Float64(liip_norm))
        )

        sol_r = solve(prob_r64, Tsit5();
                      saveat = Float64.(d.t_s),
                      reltol = 1e-6, abstol = 1e-8, verbose = false)

        # NN-augmented solve (Float32). Closure captures spl_s (per-shot
        # spline) and θ_trained (best-val checkpoint set after training).
        function dudt_eval(du, u, θ, t)
            Li, Ip, V = u
            Vind = T(Zygote.ignore(() -> spl_s(t)))
            du[1] = (-2*Vind - 2*V) / (liip_norm * softabs(Ip))
            du[2] = ( 2*Vind +    V) / (liip_norm * softabs(Li))
            ps   = re(θ_trained)
            x    = reshape(T[Li, Ip, V, Vind], 4, 1)
            y, _ = Lux.apply(romero_NN, x, ps, st_nn)
            du[3] = -V/τ - (κ_empirical/τ)*Vind + y[1]
        end

        prob_n = ODEProblem(dudt_eval, d.u0_s, (d.t_s[1], d.t_s[end]), θ_trained)
        sol_n = solve(prob_n, Tsit5();
                      p = θ_trained, saveat = d.t_s,
                      reltol = 1f-4, abstol = 1f-6, verbose = false)

        if sol_r.retcode == ReturnCode.Success && sol_n.retcode == ReturnCode.Success
            r_pred = Array(sol_r)
            n_pred = Array(sol_n)
            # Per-shot MAPE: mean over timesteps of |residual| / |target|.
            push!(romero_li_errors, mean(abs.(d.li    .- r_pred[1, :]) ./ abs.(d.li)))
            push!(romero_ip_errors, mean(abs.(d.ip_MA .- r_pred[2, :]) ./ abs.(d.ip_MA)))
            push!(nn_li_errors,     mean(abs.(d.li    .- n_pred[1, :]) ./ abs.(d.li)))
            push!(nn_ip_errors,     mean(abs.(d.ip_MA .- n_pred[2, :]) ./ abs.(d.ip_MA)))
        else
            sol_r.retcode != ReturnCode.Success && (n_failed_romero += 1)
            sol_n.retcode != ReturnCode.Success && (n_failed_nn     += 1)
        end
    end

    println("\n=== $split_name split: $(length(nn_li_errors)) shots evaluated ===")
    println("Romero Li MAPE:  $(round(100*mean(romero_li_errors), digits=3))% ± $(round(100*std(romero_li_errors), digits=3))%")
    println("Romero Ip MAPE:  $(round(100*mean(romero_ip_errors), digits=3))% ± $(round(100*std(romero_ip_errors), digits=3))%")
    println("NN     Li MAPE:  $(round(100*mean(nn_li_errors),     digits=3))% ± $(round(100*std(nn_li_errors),     digits=3))%")
    println("NN     Ip MAPE:  $(round(100*mean(nn_ip_errors),     digits=3))% ± $(round(100*std(nn_ip_errors),     digits=3))%")
    if n_failed_romero > 0 || n_failed_nn > 0
        println("Failures: Romero $n_failed_romero, NN $n_failed_nn")
    end

    return (
        romero_li = romero_li_errors,
        romero_ip = romero_ip_errors,
        nn_li     = nn_li_errors,
        nn_ip     = nn_ip_errors,
    )
end

# ============================================================================
# RUN PER-SPLIT EVALUATION
# Reports MAPE on Li and Ip for both pure Romero and the trained NN, on
# each of train/val/test splits. The TEST split is the apples-to-apples
# comparison number against Wang's Table 1.
# ============================================================================
results_train = evaluate_split(train_shot_ids, "TRAIN")
results_val   = evaluate_split(val_shot_ids,   "VAL")
results_test  = evaluate_split(test_shot_ids,  "TEST")


# ============================================================================
# DIAGNOSTIC SHOT — DETAILED CSV AND PLOT EXPORT
#
# For one chosen shot (1050218005, fixed at the top of the script), this
# block runs both the pure-Romero baseline and the trained NN forward,
# saves a side-by-side CSV of data/Romero/NN trajectories, and writes
# three comparison plots (Li, Ip, V). These artifacts are the visual
# debugging output examined across script versions.
#
# Note: training does NOT use this shot or these single-shot variables.
# This block consumes `t_shot`, `li_shot`, `ip_shot`, `V_shot` from the
# diagnostic setup at the top of the script, and `romero_trueode` and
# `prob_pred_romeroNN` defined alongside.
# ============================================================================

# Pure Romero on the diagnostic shot
romero_pred = Array(solve(romero_trueode, Tsit5();
    saveat = t_shot, reltol = 1e-6, abstol = 1e-8))

# Trained NN forward pass on the diagnostic shot
pred_full = Array(solve(prob_pred_romeroNN, Tsit5();
    p = θ_trained, saveat = t_shot,
    reltol = 1e-4, abstol = 1e-6, verbose = false))

n_solved = size(pred_full, 2)

results_df = DataFrame(
    time      = t_shot[1:n_solved],
    li_data   = li_shot[1:n_solved],
    li_romero = romero_pred[1, 1:n_solved],
    li_nn     = pred_full[1, :],
    ip_data   = ip_shot[1:n_solved],
    ip_romero = romero_pred[2, 1:n_solved],
    ip_nn     = pred_full[2, :],
    V_data    = V_shot[1:n_solved],
    V_romero  = romero_pred[3, 1:n_solved],
    V_nn      = pred_full[3, :],
)

CSV.write("predictions_RomeroNN_v62.csv", results_df)
println("Saved diagnostic predictions to predictions_RomeroNN_v62.csv")


# ----------------------------------------------------------------------------
# A separate "hard shot" CSV that exports just Li and Ip on the same shot
# but solved through a fresh NN forward pass via shot_data/shot_splines.
# Functionally similar to the above, but uses the per-shot precomputed
# data path rather than the diagnostic globals. Kept as a sanity check
# that the two paths agree.
# ----------------------------------------------------------------------------
hard_shot_id = 1050218005
d_hard       = shot_data[hard_shot_id]
spl_hard     = shot_splines[hard_shot_id]

# IMPORTANT: must stay in sign-and-scaling sync with romero_ODEfunc!,
# dudt_romeroNN_diag, dudt_shot (in loss_multishot), dudt_val (in
# val_loss), and dudt_eval (in evaluate_split).
function dudt_hard(du, u, θ, t)
    Li, Ip, V = u
    Vind = T(Zygote.ignore(() -> spl_hard(t)))
    du[1] = (-2*Vind - 2*V) / (liip_norm * softabs(Ip))
    du[2] = ( 2*Vind +    V) / (liip_norm * softabs(Li))
    ps   = re(θ_trained)
    x    = reshape(T[Li, Ip, V, Vind], 4, 1)
    y, _ = Lux.apply(romero_NN, x, ps, st_nn)
    du[3] = -V/τ - (κ_empirical/τ)*Vind + y[1]
end

prob_hard = ODEProblem(dudt_hard, d_hard.u0_s,
                       (d_hard.t_s[1], d_hard.t_s[end]), θ_trained)
sol_hard  = solve(prob_hard, Tsit5(), p = θ_trained,
                  saveat = d_hard.t_s, reltol = 1f-4, abstol = 1f-6)

hard_df = DataFrame(
    time    = d_hard.t_s,
    li_data = d_hard.li,
    li_nn   = Array(sol_hard)[1, :],
    ip_data = d_hard.ip_MA,
    ip_nn   = Array(sol_hard)[2, :],
)
CSV.write("RomeroNN_predictions_hard_shot_v62.csv", hard_df)
println("Saved hard-shot evaluation to RomeroNN_predictions_hard_shot_v62.csv")


# ============================================================================
# DIAGNOSTIC COMPARISON PLOTS
# Three figures: Li / Ip / V trajectories on the diagnostic shot, with
# data, pure-Romero baseline, and trained NN overlaid.
# ============================================================================
plt_li = plot(t_shot[1:n_solved], li_shot[1:n_solved],
    label = "Li data",   linewidth = 2, color = :blue)
plot!(plt_li, t_shot[1:n_solved], romero_pred[1, 1:n_solved],
    label = "Li Romero", linewidth = 2, linestyle = :dash, color = :green)
plot!(plt_li, t_shot[1:n_solved], pred_full[1, :],
    label = "Li NN",     linewidth = 2, linestyle = :dot,  color = :red)
title!(plt_li, "Li: data vs Romero vs NN  (shot $shot_id)")
savefig(plt_li, "li_RomeroNN_comparison_v62.png")

plt_ip = plot(t_shot[1:n_solved], ip_shot[1:n_solved],
    label = "Ip data",   linewidth = 2, color = :blue)
plot!(plt_ip, t_shot[1:n_solved], romero_pred[2, 1:n_solved],
    label = "Ip Romero", linewidth = 2, linestyle = :dash, color = :green)
plot!(plt_ip, t_shot[1:n_solved], pred_full[2, :],
    label = "Ip NN",     linewidth = 2, linestyle = :dot,  color = :red)
title!(plt_ip, "Ip: data vs Romero vs NN  (shot $shot_id)")
savefig(plt_ip, "ip_RomeroNN_comparison_v62.png")

plt_V = plot(t_shot[1:n_solved], V_shot[1:n_solved],
    label = "V data",    linewidth = 2, color = :blue)
plot!(plt_V, t_shot[1:n_solved], romero_pred[3, 1:n_solved],
    label = "V Romero",  linewidth = 2, linestyle = :dash, color = :green)
plot!(plt_V, t_shot[1:n_solved], pred_full[3, :],
    label = "V NN",      linewidth = 2, linestyle = :dot,  color = :red)
title!(plt_V, "V: data vs Romero vs NN  (shot $shot_id)")
savefig(plt_V, "V_RomeroNN_comparison_v62.png")

println("Saved comparison plots.")
