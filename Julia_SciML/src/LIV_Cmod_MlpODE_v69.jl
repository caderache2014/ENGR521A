# ============================================================================
# MlpODE_v69.jl — Pure neural ODE for plasma current and inductance
# dynamics in the Alcator C-Mod tokamak.
#
# Pure neural ODE following Wang, Garnier & Rea (2023). All three
# derivatives (dLi/dt, dIp/dt, dV/dt) are produced by a 4→32→32→3
# multilayer perceptron with no physics retained. Companion script to
# LIV_Cmod_RomeroNN_v69.jl, which implements the physics-informed
# RomeroNNV variant; comparing the two helps diagnose whether
# discrepancies vs Wang's reported numbers are model-specific or
# systematic across architectures.
#
# v69 changes vs v68:
#   1. BSON serialization of θ_trained + metadata (incl. Project.toml /
#      Manifest.toml contents) for downstream analysis.
#   2. Phil-style endpoint MAPE added alongside existing full-shot MAPE.
#   3. Full-data derivatives CSV: NN-predicted and spline-FD derivatives
#      at every measured timestep across all 489 shots, at two smoothing
#      levels (s=1e-4*N, 1e-3*N). Spline construction wrapped in safe_spline
#      with try/catch fallback (escalates s by 10× on Dierckx convergence
#      failure, up to 4 attempts) — prevents the spline-failure issue that
#      crashed RomeroNN_v69's first run.
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
using BSON                       # serialization of trained NN parameters (v69)

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
liip_norm   = T(0.85)
κ_empirical = T(0.98)
τ           = T(1.25)

# ============================================================================
# DIAGNOSTIC: NORMALIZATION STATISTICS
# 
# Run once, print, paste to chat to compare against Tino's normalization
# value of ~0.85. Identifying which combination of these stats yields 0.85
# tells us how Tino derived his normalization factor.
# ============================================================================
using Statistics

μ_li   = T(mean(df_489_shots.li))
σ_li   = T(std(df_489_shots.li))
μ_ip   = T(mean(df_489_shots.ip_MA))
σ_ip   = T(std(df_489_shots.ip_MA))
μ_V    = T(mean(df_489_shots.vc_minus_vb))
σ_V    = T(std(df_489_shots.vc_minus_vb))
μ_Vind = T(mean(df_489_shots.Vind))
σ_Vind = T(std(df_489_shots.Vind))

println("=== Per-feature stats over all 489 shots ===")
println("li:    μ=$μ_li,    σ=$σ_li")
println("ip_MA: μ=$μ_ip,    σ=$σ_ip")
println("V:     μ=$μ_V,     σ=$σ_V")
println("Vind:  μ=$μ_Vind,  σ=$σ_Vind")
println()
println("=== Candidate normalization values ===")
println("σ_ip × σ_li:                $(σ_ip * σ_li)")
println("σ_li / σ_ip:                $(σ_li / σ_ip)")
println("σ_ip / σ_li:                $(σ_ip / σ_li)")
println("mean(|ip|) × mean(|li|):    $(mean(abs.(df_489_shots.ip_MA)) * mean(abs.(df_489_shots.li)))")
println("max(|ip|) × max(|li|):      $(maximum(abs.(df_489_shots.ip_MA)) * maximum(abs.(df_489_shots.li)))   # current liip_norm")
println("σ_V × σ_Vind:               $(σ_V * σ_Vind)")
println("σ_Vind / σ_V:               $(σ_Vind / σ_V)")
println()
println("=== If we standardized each feature: stats would all be (0, 1) ===")
println("=== Looking for the value(s) closest to 0.85 ===")

# ---- dV/dt finite-difference diagnostic ----
# For each shot, compute (V[i+1] - V[i]) / (t[i+1] - t[i]).
# Pool all per-shot finite-difference estimates and report distribution stats.
# If Tino's 0.85 comes from training-data derivative scale, std() should
# land near it.
all_dVdt = Float64[]
for sid in unique(df_489_shots.shot)
    shot_rows = df_489_shots[df_489_shots.shot .== sid, :]
    t_s = shot_rows.time
    V_s = shot_rows.vc_minus_vb
    for i in 1:length(t_s)-1
        push!(all_dVdt, (V_s[i+1] - V_s[i]) / (t_s[i+1] - t_s[i]))
    end
end

println()
println("=== Finite-difference dV/dt across all 489 shots ===")
println("count:        $(length(all_dVdt))")
println("mean:         $(mean(all_dVdt))")
println("std:          $(std(all_dVdt))")
println("median:       $(median(all_dVdt))")
println("|.| mean:     $(mean(abs.(all_dVdt)))")
println("|.| std:      $(std(abs.(all_dVdt)))")
println("|.| median:   $(median(abs.(all_dVdt)))")
println("99th pctile:  $(quantile(abs.(all_dVdt), 0.99))")
println("max |dV/dt|:  $(maximum(abs.(all_dVdt)))")

# Also pool dLi/dt and dIp/dt for completeness — Tino's 0.85 might be
# derived from one of these instead of dV/dt.
all_dLidt = Float64[]
all_dIpdt = Float64[]
for sid in unique(df_489_shots.shot)
    shot_rows = df_489_shots[df_489_shots.shot .== sid, :]
    t_s  = shot_rows.time
    Li_s = shot_rows.li
    Ip_s = shot_rows.ip_MA
    for i in 1:length(t_s)-1
        push!(all_dLidt, (Li_s[i+1] - Li_s[i]) / (t_s[i+1] - t_s[i]))
        push!(all_dIpdt, (Ip_s[i+1] - Ip_s[i]) / (t_s[i+1] - t_s[i]))
    end
end

println()
println("=== Finite-difference dLi/dt and dIp/dt ===")
println("dLi/dt:  mean=$(mean(all_dLidt)),  std=$(std(all_dLidt)),  |.| mean=$(mean(abs.(all_dLidt)))")
println("dIp/dt:  mean=$(mean(all_dIpdt)),  std=$(std(all_dIpdt)),  |.| mean=$(mean(abs.(all_dIpdt)))")

# Compute the train split inline (so this diagnostic runs before the
# main split block later in the script).
#=
using Random: MersenneTwister
_diag_split_rng = MersenneTwister(42)
_diag_all_ids   = unique(df_489_shots.shot)[1:489]
_diag_shuffled  = shuffle(_diag_split_rng, copy(_diag_all_ids))
_diag_n_total   = length(_diag_shuffled)
_diag_n_train   = Int(floor(0.70 * _diag_n_total))
_diag_train_ids = _diag_shuffled[1:_diag_n_train]

train_dIpdt = Float64[]
train_dLidt = Float64[]
for sid in _diag_train_ids
    shot_rows = df_489_shots[df_489_shots.shot .== sid, :]
    t_s  = shot_rows.time
    Ip_s = shot_rows.ip_MA
    Li_s = shot_rows.li
    for i in 1:length(t_s)-1
        push!(train_dIpdt, (Ip_s[i+1] - Ip_s[i]) / (t_s[i+1] - t_s[i]))
        push!(train_dLidt, (Li_s[i+1] - Li_s[i]) / (t_s[i+1] - t_s[i]))
    end
end
println()
println("=== Train-only finite differences ($(length(_diag_train_ids)) shots) ===")
println("dLi/dt train: mean=$(mean(train_dLidt)), std=$(std(train_dLidt)), |.| mean=$(mean(abs.(train_dLidt)))")
println("dIp/dt train: mean=$(mean(train_dIpdt)), std=$(std(train_dIpdt)), |.| mean=$(mean(abs.(train_dIpdt)))")
 # Stop the script after diagnostic — don't run training, don't load NN, etc.
 =#

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
# IMPORTANT: the same three equations are inlined in `dudt_mlp_diag`
# below and inside `loss_multishot`'s local `dudt_shot`. If you change one,
# update all three.
# ============================================================================
function romero_ODEfunc!(du::AbstractVector{T}, u::AbstractVector{T}, p, t::T)
    Li, Ip, V = u
    Vind = Vind_spline32(t)
    dli_dt_denom = p.liip_norm * Ip + T(1e-8)
    dip_dt_denom = p.liip_norm * Li + T(1e-8)
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
# NEURAL NETWORK FOR THE FULL DERIVATIVE (MlpODE)
#
# Architecture: 4 → 32 → 32 → 3, following Wang et al. (2023)'s "MlpODE"
# variant. Inputs: [Li, Ip, V, Vind]. Outputs: [dLi/dt, dIp/dt, dV/dt].
# Unlike RomeroNNV which retains exact physics for du[1] and du[2], MlpODE
# replaces ALL three equations with the network. No physics baseline.
#
# Final layer uses scaled tanh (output bounded to ±5) to prevent the ODE
# solver instability we observed in early unbounded experiments.
# ============================================================================
Random.seed!(42)
rng1 = Random.default_rng()

mlp_NN = Lux.Chain(
    Lux.Dense(4, 32, softplus),
    Lux.Dense(32, 32, softplus),
    Lux.Dense(32, 3)   # 3 outputs, unbounded
)

p_nn0, st_nn = Lux.setup(rng1, mlp_NN)
Zygote.@nograd st_nn   # st_nn is empty for stateless Dense layers

# Flatten params for use with Optimization.jl, which expects a flat vector.
θ0, re = destructure(p_nn0)
θ0 = T.(θ0)

# Near-zero init of the final layer so the NN starts producing ~0 residual.
# This makes initial behavior approximately equal to pure Romero, giving
# the optimizer a sensible starting point. Final Dense(32,1) has 32
# weights + 1 bias = 33 trailing parameters in θ0.
θ0[end-98:end] .= T(0.01) .* θ0[end-98:end]   # 32×3 weights + 3 biases = 99 params


# ============================================================================
# DIAGNOSTIC NN ODE PROBLEM (TODO: relocate near comparison plots)
#
# Used only by the bottom-of-script comparison plot that runs the trained
# NN on the diagnostic shot. NOT used during training — training defines
# its own per-shot dudt closures inside loss_multishot.
# ============================================================================
function dudt_mlp_diag(du, u, θ, t)
    Li, Ip, V = u
    Vind = Vind_spline32(t)
    ps   = re(θ)
    x    = reshape(T[Li, Ip, V, Vind], 4, 1)
    y, _ = Lux.apply(mlp_NN, x, ps, st_nn)
    du[1] = y[1]
    du[2] = y[2]
    du[3] = y[3]
end

prob_pred_mlp = ODEProblem(dudt_mlp_diag, u₀, tspan32, θ0)

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
# `romero_ODEfunc!` and `dudt_mlp_diag` in sign and scaling.
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
        # and `dudt_mlp_diag` for the physics terms.
        function dudt_shot(du, u, θ, t)
                Li, Ip, V = u
                Vind = T(Zygote.ignore(() -> spl_s(t)))
                ps   = re(θ)
                x    = reshape(T[Li, Ip, V, Vind], 4, 1)
                y, _ = Lux.apply(mlp_NN, x, ps, st_nn)
                du[1] = y[1]
                du[2] = y[2]
                du[3] = y[3]
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
# loss_multishot, `dudt_mlp_diag`, and `romero_ODEfunc!`.
function val_loss(θ)
    total_loss = 0.0f0
    n_valid    = 0
    for sid in val_shot_ids
        d     = shot_data[sid]
        spl_s = shot_splines[sid]

        function dudt_val(du, u, θ, t)
            Li, Ip, V = u
            Vind = T(spl_s(Float64(t)))
            ps   = re(θ)
            x    = reshape(T[Li, Ip, V, Vind], 4, 1)
            y, _ = Lux.apply(mlp_NN, x, ps, st_nn)
            du[1] = y[1]
            du[2] = y[2]
            du[3] = y[3]
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

const total_target_iters = 7500   # scouting run; bump to 1500/5000 for full
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
    opt = Optimisers.OptimiserChain(
                        Optimisers.ClipNorm(T(1.0)),
                        Optimisers.AdamW(lr, (0.9f0, 0.999f0), T(1f-4)))
    res_chunk = Optimization.solve(optprob_chunk,
                                    opt;
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

println("Starting L-BFGS refinement phase...")
optprob_lbfgs = Optimization.OptimizationProblem(optf, res_ms.minimizer)
res_lbfgs = Optimization.solve(optprob_lbfgs,
    OptimizationOptimJL.LBFGS();
    callback = callback,
    maxiters = 500)
println("L-BFGS final loss: $(losses[end])")

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
# BSON SERIALIZATION (v69)
#
# Save the trained model parameters and the metadata needed to reproduce
# the diagnostic and downstream analyses (SR, SINDy) without retraining.
#
# What we save:
#   θ_trained        - flat parameter vector (the irreplaceable thing)
#   liip_norm        - hardcoded normalization (0.85f0)
#   train/val/test_shot_ids - exact splits used (avoids relying on seed
#                             reproducibility across Julia/package versions)
#   meta             - human-readable training config, version info, and
#                      verbatim Project.toml / Manifest.toml contents.
#
# NOTE: Unlike RomeroNN_v69, MlpODE has no κ_empirical or τ to save —
# the pure-neural model has no Romero physics constants, since all three
# derivatives come directly from the NN.
#
# What we deliberately DO NOT save:
#   re (the destructure closure) and st_nn (Lux state). These are tied to
#   the specific Lux/Flux versions at training time. The diagnostic script
#   re-defines mlp_NN with the same architecture (4 → 32 → 32 → 3,
#   softplus activations, identity output) and calls destructure() to
#   regenerate a fresh `re`. This is the version-robust pattern.
#
# Loading pattern in the diagnostic script:
#   using BSON
#   ckpt = BSON.load("mlpODE_checkpoint_v69.bson")
#   θ_trained = ckpt[:θ_trained]
#   # Re-define mlp_NN identically, then:
#   _, re = destructure(Lux.setup(rng, mlp_NN)[1])
#   ps = re(θ_trained)
# ============================================================================
const CHECKPOINT_PATH = "mlpODE_checkpoint_v69.bson"

# Helper: read a TOML file (Project.toml or Manifest.toml) from the active
# project directory. Returns the file contents as a string, or an "[error: ...]"
# placeholder if the file can't be read. Wrapped in try/catch so a missing or
# unreadable TOML file does NOT prevent the BSON checkpoint save (which would
# discard ~12 hours of training).
function _read_active_toml(filename::String)::String
    try
        proj_path = Base.active_project()
        if proj_path === nothing
            return "[error: no active project]"
        end
        proj_dir = dirname(proj_path)
        target   = joinpath(proj_dir, filename)
        if !isfile(target)
            return "[error: $filename not found in $proj_dir]"
        end
        return read(target, String)
    catch e
        return "[error reading $filename: $(sprint(showerror, e))]"
    end
end

ckpt_meta = (
    architecture     = "Chain(Dense(4,32,softplus), Dense(32,32,softplus), Dense(32,3))",
    final_train_loss = losses[end],
    best_val_loss    = best_val_loss[],
    n_val_evals      = length(val_history),
    n_train_shots    = length(train_shot_ids),
    n_val_shots      = length(val_shot_ids),
    n_test_shots     = length(test_shot_ids),
    julia_version    = string(VERSION),
    timestamp        = string(time()),
    script_version   = "v69",
    project_toml     = _read_active_toml("Project.toml"),
    manifest_toml    = _read_active_toml("Manifest.toml"),
)

BSON.@save CHECKPOINT_PATH θ_trained liip_norm train_shot_ids val_shot_ids test_shot_ids ckpt_meta
println("Saved BSON checkpoint to $(CHECKPOINT_PATH)")
println("  θ_trained length: $(length(θ_trained))")
println("  Splits: train=$(length(train_shot_ids)), val=$(length(val_shot_ids)), test=$(length(test_shot_ids))")

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
# scaling sync with `romero_ODEfunc!`, `dudt_mlp_diag`, `dudt_shot`
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
            dli_denom = p.liip_norm * Ip + 1e-8
            dip_denom = p.liip_norm * Li + 1e-8
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
            ps   = re(θ_trained)
            x    = reshape(T[Li, Ip, V, Vind], 4, 1)
            y, _ = Lux.apply(mlp_NN, x, ps, st_nn)
            du[1] = y[1]
            du[2] = y[2]
            du[3] = y[3]
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
# PHIL-STYLE ENDPOINT MAPE EVALUATION (v69 addition)
#
# Emulates the MAPE calculation in Phil's JAX RomeroNNV implementation
# (see phils_jax_romeroNN.py lines 658-668). For each shot in a split:
#   1. Segment the trajectory into non-overlapping 10-step episodes,
#      starts at [0, 10] per shot when T=28 (matches Phil exactly).
#   2. Initial condition for each episode = measured state at start.
#   3. Integrate forward 10 steps; relative error at FINAL timestep only:
#         100 * |pred[end] - target[end]| / |target[end]|
#   4. Aggregate mean ± std across all episodes from all shots in split.
#
# IMPORTANT: This evaluates Phil's METRIC on this codebase's SHOT-LEVEL
# test split. Phil's full pipeline splits EPISODES 70/15/15, which has
# shot-level data leakage; we deliberately do not replicate that.
#
# NOTE: Unlike RomeroNN_v69's evaluate_split_endpoint, the dudt_eval_phil
# closure here does NOT contain Romero physics. All three derivatives come
# directly from the NN (this is a pure neural ODE, not a hybrid model).
# ============================================================================
function evaluate_split_endpoint(shot_id_list, split_name::String;
    group_size::Int=10, stride::Int=10)
    nn_li_endpoint_errors = Float32[]
    nn_ip_endpoint_errors = Float32[]
    n_failed_nn = 0
    n_episodes_total = 0

    for sid in shot_id_list
        d = shot_data[sid]
        spl_s = shot_splines[sid]
        Tsh = length(d.t_s)

        function dudt_eval_phil(du, u, θ, t)
            Li, Ip, V = u
            Vind = T(Zygote.ignore(() -> spl_s(t)))
            ps = re(θ_trained)
            x = reshape(T[Li, Ip, V, Vind], 4, 1)
            y, _ = Lux.apply(mlp_NN, x, ps, st_nn)
            du[1] = y[1]
            du[2] = y[2]
            du[3] = y[3]
        end

        # Phil's Python range(0, Tsh - group_size, stride) -> Julia equivalent:
        start_indices_0 = 0:stride:(Tsh-group_size-1)

        for s0 in start_indices_0
            n_episodes_total += 1

            ep_idx_start = s0 + 1
            ep_idx_end = s0 + group_size
            ep_t = d.t_s[ep_idx_start:ep_idx_end]

            u0_ep = T[d.li[ep_idx_start], d.ip_MA[ep_idx_start], d.V[ep_idx_start]]

            prob_ep = ODEProblem(dudt_eval_phil, u0_ep, (ep_t[1], ep_t[end]), θ_trained)
            sol_ep = solve(prob_ep,
                            Tsit5();
                            p=θ_trained,
                            saveat=ep_t,
                            reltol=1.0f-4, 
                            abstol=1.0f-6, 
                            verbose=false)

            if sol_ep.retcode == ReturnCode.Success
                pred_ep = Array(sol_ep)
                li_target = d.li[ep_idx_end]
                ip_target = d.ip_MA[ep_idx_end]
                push!(nn_li_endpoint_errors, 100.0f0 * abs(pred_ep[1, end] - li_target) / abs(li_target))
                push!(nn_ip_endpoint_errors, 100.0f0 * abs(pred_ep[2, end] - ip_target) / abs(ip_target))
            else
                n_failed_nn += 1
            end
        end
    end

    n_evaluated = length(nn_li_endpoint_errors)
    println("\n=== $split_name endpoint MAPE (Phil-style, group_size=$group_size, stride=$stride) ===")
    println("Episodes attempted:  $n_episodes_total")
    println("Episodes evaluated:  $n_evaluated")
    if n_failed_nn > 0
        println("Episodes failed:     $n_failed_nn")
    end
    if n_evaluated > 0
        println("NN Li endpoint MAPE: $(round(mean(nn_li_endpoint_errors), digits=3))% ± $(round(std(nn_li_endpoint_errors), digits=3))%")
        println("NN Ip endpoint MAPE: $(round(mean(nn_ip_endpoint_errors), digits=3))% ± $(round(std(nn_ip_endpoint_errors), digits=3))%")
    end

    return (
        nn_li_endpoint=nn_li_endpoint_errors,
        nn_ip_endpoint=nn_ip_endpoint_errors,
        n_episodes=n_episodes_total,
        n_failed=n_failed_nn,
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

results_train_endpoint = evaluate_split_endpoint(train_shot_ids, "TRAIN")
results_val_endpoint   = evaluate_split_endpoint(val_shot_ids,   "VAL")
results_test_endpoint  = evaluate_split_endpoint(test_shot_ids,  "TEST")

# Save raw per-shot errors (full-shot MAPE) for paired analysis.
fullshot_test_df = DataFrame(
    shot_id   = test_shot_ids[1:length(results_test.nn_li)],
    nn_li     = results_test.nn_li,
    nn_ip     = results_test.nn_ip,
    romero_li = results_test.romero_li,
    romero_ip = results_test.romero_ip,
)
CSV.write("fullshot_mape_TEST_MlpODE_v69.csv", fullshot_test_df)
println("Saved per-shot full-shot MAPE (TEST) to fullshot_mape_TEST_MlpODE_v69.csv")

# Save raw per-episode errors (endpoint MAPE) for paired analysis.
endpoint_test_df = DataFrame(
    nn_li_endpoint = results_test_endpoint.nn_li_endpoint,
    nn_ip_endpoint = results_test_endpoint.nn_ip_endpoint,
)
CSV.write("endpoint_mape_TEST_MlpODE_v69.csv", endpoint_test_df)
println("Saved per-episode endpoint MAPE (TEST) to endpoint_mape_TEST_MlpODE_v69.csv")

# ============================================================================
# FULL-DATA DERIVATIVES EXTRACTION (v69 addition)
#
# For every measured timestep across all 489 shots, save:
#   - Metadata: shot_id, time, split, is_interior flag
#   - State and control: Li, Ip, V, Vind (Vind = spline value at t_s[i],
#     matching what the NN sees at training/eval time)
#   - NN-predicted derivatives at the measured (Li, Ip, V, Vind) point.
#     Unlike RomeroNN_v69, all three derivatives here come from the NN
#     directly (this is a pure neural ODE, no Romero physics).
#   - Finite-difference derivatives from data, computed via Dierckx splines
#     at two smoothing levels: s=1e-4*N (light) and s=1e-3*N (matches Vind).
#     Two FD columns per channel × three channels = 6 FD columns.
#
# Powers: V_dot diagnostic, SR Scenario B prep, SINDy on any channel,
# per-shot breakdown.
#
# Spline robustness: safe_spline wraps Dierckx Spline1D in a try/catch
# that escalates s by 10× on convergence failure, up to 4 attempts. This
# is necessary because RomeroNN_v69 hit "maxit reached, s too small"
# errors on a small fraction of shots (~3 / 489) at s=1e-4*N. The fallback
# is logged so we know which shots required it.
# ============================================================================
println("\n=== Building full-data derivatives CSV across all 489 shots ===")

split_lookup = Dict{Int64, String}()
for sid in train_shot_ids; split_lookup[sid] = "train"; end
for sid in val_shot_ids;   split_lookup[sid] = "val";   end
for sid in test_shot_ids;  split_lookup[sid] = "test";  end

# Pre-instantiate parameters once (avoids ~14k redundant re() calls).
ps_trained = re(θ_trained)

# safe_spline: try requested s; on Dierckx failure escalate s by 10× and retry,
# up to 4 attempts. Logs which shots/channels required fallback.
function safe_spline(t, y, s_target, sid, channel_name)
    s_try = s_target
    for attempt in 1:4
        try
            return Spline1D(t, y; k=3, bc="extrapolate", s=s_try)
        catch e
            if attempt < 4
                println("  shot $sid: $channel_name spline failed at s=$(s_try), escalating to s=$(s_try*10)")
                s_try *= 10
            else
                rethrow(e)
            end
        end
    end
end

n_rows_estimate = sum(length(shot_data[sid].t_s) for sid in shot_ids)
col_shot         = Vector{Int64}(undef,   0); sizehint!(col_shot, n_rows_estimate)
col_time         = Vector{Float32}(undef, 0); sizehint!(col_time, n_rows_estimate)
col_split        = Vector{String}(undef,  0); sizehint!(col_split, n_rows_estimate)
col_interior     = Vector{Bool}(undef,    0); sizehint!(col_interior, n_rows_estimate)
col_Li           = Vector{Float32}(undef, 0); sizehint!(col_Li, n_rows_estimate)
col_Ip           = Vector{Float32}(undef, 0); sizehint!(col_Ip, n_rows_estimate)
col_V            = Vector{Float32}(undef, 0); sizehint!(col_V, n_rows_estimate)
col_Vind         = Vector{Float32}(undef, 0); sizehint!(col_Vind, n_rows_estimate)
col_Li_dot_NN    = Vector{Float32}(undef, 0); sizehint!(col_Li_dot_NN, n_rows_estimate)
col_Ip_dot_NN    = Vector{Float32}(undef, 0); sizehint!(col_Ip_dot_NN, n_rows_estimate)
col_V_dot_NN     = Vector{Float32}(undef, 0); sizehint!(col_V_dot_NN, n_rows_estimate)
col_Li_dot_s1em4 = Vector{Float32}(undef, 0); sizehint!(col_Li_dot_s1em4, n_rows_estimate)
col_Li_dot_s1em3 = Vector{Float32}(undef, 0); sizehint!(col_Li_dot_s1em3, n_rows_estimate)
col_Ip_dot_s1em4 = Vector{Float32}(undef, 0); sizehint!(col_Ip_dot_s1em4, n_rows_estimate)
col_Ip_dot_s1em3 = Vector{Float32}(undef, 0); sizehint!(col_Ip_dot_s1em3, n_rows_estimate)
col_V_dot_s1em4  = Vector{Float32}(undef, 0); sizehint!(col_V_dot_s1em4, n_rows_estimate)
col_V_dot_s1em3  = Vector{Float32}(undef, 0); sizehint!(col_V_dot_s1em3, n_rows_estimate)

n_shots_done = 0
for sid in shot_ids
    d     = shot_data[sid]
    spl_v = shot_splines[sid]   # Vind spline (s=1e-3*N), matches NN training input
    Tsh   = length(d.t_s)
    sp    = split_lookup[sid]

    Nf = Float64(Tsh)

    # FD splines for all three channels, two smoothing levels each, with fallback.
    spl_Li_s1em4 = safe_spline(d.t_s, d.li,    1e-4 * Nf, sid, "Li_s1em4")
    spl_Li_s1em3 = safe_spline(d.t_s, d.li,    1e-3 * Nf, sid, "Li_s1em3")
    spl_Ip_s1em4 = safe_spline(d.t_s, d.ip_MA, 1e-4 * Nf, sid, "Ip_s1em4")
    spl_Ip_s1em3 = safe_spline(d.t_s, d.ip_MA, 1e-3 * Nf, sid, "Ip_s1em3")
    spl_V_s1em4  = safe_spline(d.t_s, d.V,     1e-4 * Nf, sid, "V_s1em4")
    spl_V_s1em3  = safe_spline(d.t_s, d.V,     1e-3 * Nf, sid, "V_s1em3")

    for i in 1:Tsh
        t_i    = d.t_s[i]
        Li_i   = d.li[i]
        Ip_i   = d.ip_MA[i]
        V_i    = d.V[i]
        Vind_i = T(spl_v(t_i))

        # NN forward pass: all three derivatives come directly from the NN.
        x_in      = reshape(T[Li_i, Ip_i, V_i, Vind_i], 4, 1)
        nn_out, _ = Lux.apply(mlp_NN, x_in, ps_trained, st_nn)

        Li_dot_NN_i = nn_out[1]
        Ip_dot_NN_i = nn_out[2]
        V_dot_NN_i  = nn_out[3]

        push!(col_shot,         sid)
        push!(col_time,         t_i)
        push!(col_split,        sp)
        push!(col_interior,     i != 1 && i != Tsh)
        push!(col_Li,           Li_i)
        push!(col_Ip,           Ip_i)
        push!(col_V,            V_i)
        push!(col_Vind,         Vind_i)
        push!(col_Li_dot_NN,    Li_dot_NN_i)
        push!(col_Ip_dot_NN,    Ip_dot_NN_i)
        push!(col_V_dot_NN,     V_dot_NN_i)
        push!(col_Li_dot_s1em4, T(derivative(spl_Li_s1em4, t_i)))
        push!(col_Li_dot_s1em3, T(derivative(spl_Li_s1em3, t_i)))
        push!(col_Ip_dot_s1em4, T(derivative(spl_Ip_s1em4, t_i)))
        push!(col_Ip_dot_s1em3, T(derivative(spl_Ip_s1em3, t_i)))
        push!(col_V_dot_s1em4,  T(derivative(spl_V_s1em4,  t_i)))
        push!(col_V_dot_s1em3,  T(derivative(spl_V_s1em3,  t_i)))
    end

    global n_shots_done += 1
    if n_shots_done % 50 == 0
        println("  Processed $n_shots_done / $(length(shot_ids)) shots")
    end
end

derivatives_df = DataFrame(
    shot_id         = col_shot,
    time            = col_time,
    split           = col_split,
    is_interior     = col_interior,
    Li              = col_Li,
    Ip              = col_Ip,
    V               = col_V,
    Vind            = col_Vind,
    Li_dot_NN       = col_Li_dot_NN,
    Ip_dot_NN       = col_Ip_dot_NN,
    V_dot_NN        = col_V_dot_NN,
    Li_dot_FD_s1em4 = col_Li_dot_s1em4,
    Li_dot_FD_s1em3 = col_Li_dot_s1em3,
    Ip_dot_FD_s1em4 = col_Ip_dot_s1em4,
    Ip_dot_FD_s1em3 = col_Ip_dot_s1em3,
    V_dot_FD_s1em4  = col_V_dot_s1em4,
    V_dot_FD_s1em3  = col_V_dot_s1em3,
)
CSV.write("derivatives_MlpODE_v69.csv", derivatives_df)
println("Saved derivatives CSV: $(nrow(derivatives_df)) rows × $(ncol(derivatives_df)) cols → derivatives_MlpODE_v69.csv")

# ============================================================================
# SAVE TRAINING / VALIDATION LOSS HISTORY
#
# losses[] is appended every AdamW iteration by the callback (full
# trajectory). val_history[] is appended every n_epochs_per_val=50
# iterations as (iter, val_loss) tuples. Save both to CSV for offline
# plotting and for comparison with Wang's published loss curves.
# ============================================================================
loss_df = DataFrame(
    iter       = 1 : length(losses),
    train_loss = losses,
)
CSV.write("loss_history_MlpODE_v69.csv", loss_df)
println("Saved training loss history to loss_history_MlpODE_v69.csv")

val_df = DataFrame(
    iter     = [vh[1] for vh in val_history],
    val_loss = [vh[2] for vh in val_history],
)
CSV.write("val_loss_history_MlpODE_v68.csv", val_df)
println("Saved validation loss history to val_loss_history_MlpODE_v69.csv")

# Quick diagnostic plot (PNG). Logarithmic y-axis since the loss
# spans orders of magnitude during training.
plt_loss = plot(loss_df.iter, loss_df.train_loss,
                yscale = :log10,
                xlabel = "Iteration",
                ylabel = "Loss (log scale)",
                label  = "Train",
                title  = "Training & Validation Loss vs Iteration",
                lw     = 1.5,
                color  = :steelblue)
plot!(plt_loss, val_df.iter, val_df.val_loss,
      label = "Validation",
      lw    = 2,
      color = :crimson,
      seriestype = :scatter,
      ms    = 4)
savefig(plt_loss, "loss_history_MlpODE_v69.png")
println("Saved loss-history plot to loss_history_MlpODE_v69.png")

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
# `prob_pred_mlp` defined alongside.
# ============================================================================

# Pure Romero on the diagnostic shot
romero_pred = Array(solve(romero_trueode, Tsit5();
    saveat = t_shot, reltol = 1e-6, abstol = 1e-8))

# Trained NN forward pass on the diagnostic shot
pred_full = Array(solve(prob_pred_mlp, Tsit5();
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

CSV.write("predictions_MlpODE_v69.csv", results_df)
println("Saved diagnostic predictions to predictions_MlpODE_v69.csv")


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
# dudt_mlp_diag, dudt_shot (in loss_multishot), dudt_val (in
# val_loss), and dudt_eval (in evaluate_split).
function dudt_hard(du, u, θ, t)
    Li, Ip, V = u
    Vind = T(Zygote.ignore(() -> spl_hard(t)))
    ps   = re(θ_trained)
    x    = reshape(T[Li, Ip, V, Vind], 4, 1)
    y, _ = Lux.apply(mlp_NN, x, ps, st_nn)
    du[1] = y[1]
    du[2] = y[2]
    du[3] = y[3]
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
CSV.write("MlpODE_predictions_hard_shot_v69.csv", hard_df)
println("Saved hard-shot evaluation to MlpODE_predictions_hard_shot_v69.csv")


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
savefig(plt_li, "li_MlpODE_comparison_v69.png")

plt_ip = plot(t_shot[1:n_solved], ip_shot[1:n_solved],
    label = "Ip data",   linewidth = 2, color = :blue)
plot!(plt_ip, t_shot[1:n_solved], romero_pred[2, 1:n_solved],
    label = "Ip Romero", linewidth = 2, linestyle = :dash, color = :green)
plot!(plt_ip, t_shot[1:n_solved], pred_full[2, :],
    label = "Ip NN",     linewidth = 2, linestyle = :dot,  color = :red)
title!(plt_ip, "Ip: data vs Romero vs NN  (shot $shot_id)")
savefig(plt_ip, "ip_MlpODE_comparison_v69.png")

plt_V = plot(t_shot[1:n_solved], V_shot[1:n_solved],
    label = "V data",    linewidth = 2, color = :blue)
plot!(plt_V, t_shot[1:n_solved], romero_pred[3, 1:n_solved],
    label = "V Romero",  linewidth = 2, linestyle = :dash, color = :green)
plot!(plt_V, t_shot[1:n_solved], pred_full[3, :],
    label = "V NN",      linewidth = 2, linestyle = :dot,  color = :red)
title!(plt_V, "V: data vs Romero vs NN  (shot $shot_id)")
savefig(plt_V, "V_MlpODE_comparison_v69.png")

println("Saved comparison plots.")

