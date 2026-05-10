# ============================================================================
# recover_romeroNN_v70.jl
#
# Recovery script for RomeroNN_v70 outputs.
#
# CONTEXT: LIV_Cmod_RomeroNN_v70.jl crashed during the FULL-DATA DERIVATIVES
# EXTRACTION block when Dierckx.Spline1D failed on a noisy V channel for
# some shot. The crash happened AFTER training completed and AFTER the BSON
# checkpoint and MAPE CSVs were written, but BEFORE the derivatives CSV,
# single-shot predictions, hard-shot predictions, or loss history files
# were saved.
#
# This script reconstructs all of those except the loss-history values at
# every iteration, which were held in memory and lost. The terminal log
# captured every-50-iters, which we parse here for a coarser reconstruction.
#
# Inputs (in wang_model_workspace/):
#   romeroNN_checkpoint_v70.bson      — trained NN parameters and metadata
#   src/romero_shots_489/*.csv        — original 489 shot CSVs (same as v70)
#   romeroNN_v70_run_<TIMESTAMP>.log  — terminal log for loss history parsing
#
# Outputs (written to wang_model_workspace/):
#   derivatives_RomeroNN_v70.csv               — for V_dot diagnostic
#   predictions_RomeroNN_v70.csv               — single-shot trajectories
#   RomeroNN_predictions_hard_shot_v70.csv     — hard-shot Li/Ip subset
#   loss_history_RomeroNN_v70.csv              — every-50-iter train losses
#   val_loss_history_RomeroNN_v70.csv          — every-50-iter val losses
#
# Run from src/:
#   julia --project=.. recover_romeroNN_v70.jl 2>&1 | tee ../recover_romeroNN_v70.log
# ============================================================================

using BSON
using CSV
using DataFrames
using Dierckx
using Glob
using Lux
using Optimisers
using OrdinaryDiffEq
using Random
using SciMLSensitivity
using Statistics
using Zygote

const T = Float32

# ============================================================================
# CONFIG — paths and identifiers (match v70's hardcoded values)
# ============================================================================
# This recovery script is run from src/, so paths are relative to wang_model_workspace.
const CHECKPOINT_PATH = "../romeroNN_checkpoint_v70.bson"
const SHOTS_DIR       = "romero_shots_489"   # relative to src/, matches v70
const LOG_PATH        = "../romeroNN_v70_run_20260508_1804.log"  # update if filename differs

const DERIVATIVES_OUT      = "../derivatives_RomeroNN_v70.csv"
const PREDICTIONS_OUT      = "../predictions_RomeroNN_v70.csv"
const HARDSHOT_OUT         = "../RomeroNN_predictions_hard_shot_v70.csv"
const LOSS_HISTORY_OUT     = "../loss_history_RomeroNN_v70.csv"
const VAL_LOSS_HISTORY_OUT = "../val_loss_history_RomeroNN_v70.csv"

const HARD_SHOT_ID = 1050218005

# ============================================================================
# 1. LOAD CHECKPOINT
# ============================================================================
println("=== Loading checkpoint $CHECKPOINT_PATH ===")
ckpt = BSON.load(CHECKPOINT_PATH)
θ_trained      = ckpt[:θ_trained]
liip_norm      = ckpt[:liip_norm]
κ_empirical    = ckpt[:κ_empirical]
τ              = ckpt[:τ]
train_shot_ids = ckpt[:train_shot_ids]
val_shot_ids   = ckpt[:val_shot_ids]
test_shot_ids  = ckpt[:test_shot_ids]
ckpt_meta      = ckpt[:ckpt_meta]

println("  θ_trained length: $(length(θ_trained))")
println("  liip_norm = $liip_norm, κ_empirical = $κ_empirical, τ = $τ")
println("  Splits: train=$(length(train_shot_ids)), val=$(length(val_shot_ids)), test=$(length(test_shot_ids))")

# ============================================================================
# 2. REBUILD THE TRAINED NN
# ============================================================================
println("\n=== Rebuilding trained NN ===")
Random.seed!(42)
rng1 = Random.default_rng()

romero_NN = Lux.Chain(
    Lux.Dense(4, 32, softplus),
    Lux.Dense(32, 32, softplus),
    Lux.Dense(32, 1)
)

p_nn0, st_nn = Lux.setup(rng1, romero_NN)
Zygote.@nograd st_nn
θ0, re = destructure(p_nn0)
ps_trained = re(θ_trained)
println("  NN reconstructed; ready for inference")

# ============================================================================
# 3. LOAD DATA (mirror v70's data loading exactly)
# ============================================================================
println("\n=== Loading data from $SHOTS_DIR ===")
csv_files    = glob("*.csv", SHOTS_DIR)
df_list      = DataFrame.(CSV.File.(csv_files))
df_489_shots = reduce(vcat, df_list)
shot_ids     = unique(df_489_shots.shot)
println("  Loaded $(length(shot_ids)) shots, $(nrow(df_489_shots)) total rows")

# Build shot_splines and shot_data exactly as v70 does (lines 396-416 of v70).
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
        t_s   = t_s,
        u0_s  = T[shot.li[1], shot.ip_MA[1], shot.vc_minus_vb[1]],
        li    = T.(shot.li),
        ip_MA = T.(shot.ip_MA),
        V     = T.(shot.vc_minus_vb),
    )
end
println("  Built shot_data and shot_splines for $(length(shot_data)) shots")

split_lookup = Dict{Int64, String}()
for sid in train_shot_ids; split_lookup[sid] = "train"; end
for sid in val_shot_ids;   split_lookup[sid] = "val";   end
for sid in test_shot_ids;  split_lookup[sid] = "test";  end

# ============================================================================
# 4. SAFE SPLINE WRAPPER (the missing piece from v70)
# ============================================================================
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

# ============================================================================
# 5. DERIVATIVES CSV — the main goal for V_dot diagnostic
# ============================================================================
println("\n=== Building $DERIVATIVES_OUT ===")

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
    spl_v = shot_splines[sid]
    Tsh   = length(d.t_s)
    sp    = split_lookup[sid]

    Nf = Float64(Tsh)

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

        # NN forward pass — RomeroNN's NN produces a SCALAR residual added to V_dot.
        # Li_dot and Ip_dot come from pure Romero physics (no NN involvement).
        x_in      = reshape(T[Li_i, Ip_i, V_i, Vind_i], 4, 1)
        nn_out, _ = Lux.apply(romero_NN, x_in, ps_trained, st_nn)

        Li_dot_phys = (-2*Vind_i - 2*V_i) / (liip_norm * Ip_i + T(1e-8))
        Ip_dot_phys = ( 2*Vind_i +    V_i) / (liip_norm * Li_i + T(1e-8))
        V_dot_total = -V_i/τ - (κ_empirical/τ)*Vind_i + nn_out[1]

        push!(col_shot,         sid)
        push!(col_time,         t_i)
        push!(col_split,        sp)
        push!(col_interior,     i != 1 && i != Tsh)
        push!(col_Li,           Li_i)
        push!(col_Ip,           Ip_i)
        push!(col_V,            V_i)
        push!(col_Vind,         Vind_i)
        push!(col_Li_dot_NN,    Li_dot_phys)
        push!(col_Ip_dot_NN,    Ip_dot_phys)
        push!(col_V_dot_NN,     V_dot_total)
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
CSV.write(DERIVATIVES_OUT, derivatives_df)
println("  Saved $(nrow(derivatives_df)) rows × $(ncol(derivatives_df)) cols → $DERIVATIVES_OUT")

# ============================================================================
# 6. SINGLE-SHOT TRAJECTORY (predictions_RomeroNN_v70.csv)
# ============================================================================
println("\n=== Building $PREDICTIONS_OUT (shot $HARD_SHOT_ID) ===")

d_diag    = shot_data[HARD_SHOT_ID]
spl_diag  = shot_splines[HARD_SHOT_ID]
t_shot    = d_diag.t_s
li_shot   = d_diag.li
ip_shot   = d_diag.ip_MA
V_shot    = d_diag.V
u0_diag   = d_diag.u0_s

function dudt_romeroNN(du, u, θ, t)
    Li, Ip, V = u
    Vind = T(spl_diag(Float64(t)))
    du[1] = (-2*Vind - 2*V) / (liip_norm * Ip + T(1e-8))
    du[2] = ( 2*Vind +    V) / (liip_norm * Li + T(1e-8))
    ps   = re(θ)
    x    = reshape(T[Li, Ip, V, Vind], 4, 1)
    y, _ = Lux.apply(romero_NN, x, ps, st_nn)
    du[3] = -V/τ - (κ_empirical/τ)*Vind + y[1]
end

function dudt_romero_only(du, u, θ, t)
    Li, Ip, V = u
    Vind = T(spl_diag(Float64(t)))
    du[1] = (-2*Vind - 2*V) / (liip_norm * Ip + T(1e-8))
    du[2] = ( 2*Vind +    V) / (liip_norm * Li + T(1e-8))
    du[3] = -V/τ - (κ_empirical/τ)*Vind
end

prob_nn = ODEProblem(dudt_romeroNN, u0_diag, (t_shot[1], t_shot[end]), θ_trained)
sol_nn  = solve(prob_nn, Tsit5(); p = θ_trained, saveat = t_shot,
                reltol = 1f-4, abstol = 1f-6, verbose = false)
pred_full = Array(sol_nn)

prob_rom = ODEProblem(dudt_romero_only, u0_diag, (t_shot[1], t_shot[end]), θ_trained)
sol_rom  = solve(prob_rom, Tsit5(); p = θ_trained, saveat = t_shot,
                 reltol = 1f-4, abstol = 1f-6, verbose = false)
romero_pred = Array(sol_rom)

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
CSV.write(PREDICTIONS_OUT, results_df)
println("  Saved $(nrow(results_df)) timesteps → $PREDICTIONS_OUT")

# ============================================================================
# 7. HARD-SHOT CSV (Li and Ip only — same shot, smaller schema)
# ============================================================================
println("\n=== Building $HARDSHOT_OUT ===")

hard_df = DataFrame(
    time    = t_shot[1:n_solved],
    li_data = li_shot[1:n_solved],
    li_nn   = pred_full[1, :],
    ip_data = ip_shot[1:n_solved],
    ip_nn   = pred_full[2, :],
)
CSV.write(HARDSHOT_OUT, hard_df)
println("  Saved $(nrow(hard_df)) timesteps → $HARDSHOT_OUT")

# ============================================================================
# 8. PARSE LOG TO RECONSTRUCT LOSS HISTORIES
# ============================================================================
println("\n=== Parsing $LOG_PATH for loss histories ===")

log_lines = readlines(LOG_PATH)

train_iters  = Int[]
train_losses = Float64[]
val_iters    = Int[]
val_losses   = Float64[]

for line in log_lines
    m_train = match(r"^Iter (\d+): train_loss = ([\d.e+\-]+)", line)
    if m_train !== nothing
        push!(train_iters, parse(Int, m_train.captures[1]))
        push!(train_losses, parse(Float64, m_train.captures[2]))
        continue
    end
    m_val = match(r"^\s*Iter (\d+): val_loss = ([\d.e+\-]+)", line)
    if m_val !== nothing
        push!(val_iters, parse(Int, m_val.captures[1]))
        push!(val_losses, parse(Float64, m_val.captures[2]))
    end
end

println("  Parsed $(length(train_iters)) train_loss rows, $(length(val_iters)) val_loss rows")

train_df = DataFrame(iter = train_iters, train_loss = train_losses)
val_df   = DataFrame(iter = val_iters,   val_loss   = val_losses)
CSV.write(LOSS_HISTORY_OUT,     train_df)
CSV.write(VAL_LOSS_HISTORY_OUT, val_df)
println("  Saved → $LOSS_HISTORY_OUT")
println("  Saved → $VAL_LOSS_HISTORY_OUT")
println("  NOTE: granularity is every-50-iters (vs every-iter for v68/v69 originals).")
println("  When plotting v69 vs v70 comparisons, down-sample v69 to match.")

println("\n=== Recovery complete. ===")