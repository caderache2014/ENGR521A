# ============================================================================
# vdot_diagnostic_MlpODE_v70.jl
#
# MlpODE_v70 version of the V_dot correlation diagnostic. 
#
# Hardcoded τ = 1.25 and κ_empirical = 0.98 — physics constants from
# Wang et al. (2023), unchanged across script versions.
#
# Inputs (in wang_model_workspace/):
#   derivatives_MlpODE_v70.csv   
#
# Outputs (written to wang_model_workspace/):
#   vdot_diagnostic_summary_MlpODE_v70.csv
#   vdot_diagnostic_perpoint_MlpODE_v70.csv
#   vdot_diagnostic_pershot_MlpODE_v70.csv
#   vdot_pathological_shots_MlpODE_v70.csv
#
# Run from src/:
#   julia --project=.. vdot_diagnostic_MlpODE_v70.jl 2>&1 | tee ../vdot_diagnostic_MlpODE_v70.log
# ============================================================================

using CSV
using DataFrames
using Statistics
using StatsBase
using Printf

const T = Float32

# ============================================================================
# CONFIG
# ============================================================================
const DERIVATIVES_CSV = "../derivatives_MlpODE_v70.csv"

const SUMMARY_OUT      = "../vdot_diagnostic_summary_MlpODE_v70.csv"
const PERPOINT_OUT     = "../vdot_diagnostic_perpoint_MlpODE_v70.csv"
const PERSHOT_OUT      = "../vdot_diagnostic_pershot_MlpODE_v70.csv"
const PATHOLOGICAL_OUT = "../vdot_pathological_shots_MlpODE_v70.csv"

const CLIP_FRACTION          = 0.99
const HEADLINE_SMOOTHING      = "s1em3"
const PATHOLOGICAL_THRESHOLD = 5

# Romero physics constants (Wang et al. 2023; identical across all script versions).
const τ           = 1.25
const κ_empirical = 0.98

# ============================================================================
# LOAD DATA
# ============================================================================
println("Loading $DERIVATIVES_CSV ...")
df = CSV.read(DERIVATIVES_CSV, DataFrame)
println("  $(nrow(df)) rows × $(ncol(df)) cols")
println("  Splits: $(combine(groupby(df, :split), nrow => :n))")
println("  Romero physics constants (hardcoded): τ = $τ, κ_empirical = $κ_empirical")

# ============================================================================
# COMPUTE PURE-ROMERO V_dot BASELINE
# ============================================================================
println("\nComputing pure-Romero V_dot baseline ...")
df.V_dot_Romero = .-df.V ./ τ .- (κ_empirical / τ) .* df.Vind

# ============================================================================
# UTILITIES
# ============================================================================
function compute_stats(pred::AbstractVector, target::AbstractVector)
    n = length(pred)
    if n < 2
        return (n=n, R2=NaN, Pearson=NaN, Spearman=NaN, RMSE=NaN)
    end
    r_p  = cor(pred, target)
    r_s  = corspearman(pred, target)
    rmse = sqrt(mean((pred .- target).^2))
    return (n=n, R2=r_p^2, Pearson=r_p, Spearman=r_s, RMSE=rmse)
end

function middle_fraction_mask(reference::AbstractVector, keep_fraction::Float64)
    n = length(reference)
    drop_each_side = floor(Int, n * (1 - keep_fraction) / 2)
    if drop_each_side == 0
        return trues(n)
    end
    sorted_abs = sort(abs.(reference))
    lo_thresh = sorted_abs[drop_each_side + 1]
    hi_thresh = sorted_abs[n - drop_each_side]
    return (abs.(reference) .>= lo_thresh) .& (abs.(reference) .<= hi_thresh)
end

# ============================================================================
# BUILD SUMMARY TABLE
# ============================================================================
println("\nComputing summary statistics ...")

summary_rows = NamedTuple[]

for split_name in ["train", "val", "test"]
    df_sp = df[df.split .== split_name, :]

    for (smoothing_label, fd_col) in [("s1em4", :V_dot_FD_s1em4), ("s1em3", :V_dot_FD_s1em3)]
        for clipping_label in ["raw", "middle99"]
            for point_filter_label in ["all", "interior_only"]
                if point_filter_label == "interior_only"
                    df_pf = df_sp[df_sp.is_interior, :]
                else
                    df_pf = df_sp
                end

                if clipping_label == "middle99"
                    mask = middle_fraction_mask(df_pf[!, fd_col], CLIP_FRACTION)
                    df_clip = df_pf[mask, :]
                else
                    df_clip = df_pf
                end

                vdot_fd     = df_clip[!, fd_col]
                vdot_nn     = df_clip.V_dot_NN
                vdot_romero = df_clip.V_dot_Romero

                stats_nn     = compute_stats(vdot_nn,     vdot_fd)
                stats_romero = compute_stats(vdot_romero, vdot_fd)

                push!(summary_rows, (
                    split          = split_name,
                    smoothing      = smoothing_label,
                    clipping       = clipping_label,
                    point_filter   = point_filter_label,
                    n_points       = stats_nn.n,
                    R2_NN          = stats_nn.R2,
                    Pearson_NN     = stats_nn.Pearson,
                    Spearman_NN    = stats_nn.Spearman,
                    RMSE_NN        = stats_nn.RMSE,
                    R2_Romero      = stats_romero.R2,
                    Pearson_Romero = stats_romero.Pearson,
                    Spearman_Romero= stats_romero.Spearman,
                    RMSE_Romero    = stats_romero.RMSE,
                    R2_improvement_NN_over_Romero = stats_nn.R2 - stats_romero.R2,
                ))
            end
        end
    end
end

summary_df = DataFrame(summary_rows)
CSV.write(SUMMARY_OUT, summary_df)
println("  Saved $(nrow(summary_df)) summary rows → $SUMMARY_OUT")

# ============================================================================
# PER-SHOT STATISTICS
# ============================================================================
println("\nComputing per-shot statistics ...")

pershot_rows = NamedTuple[]

for shot_grp in groupby(df, :shot_id)
    sid       = shot_grp.shot_id[1]
    split_lbl = shot_grp.split[1]

    interior = shot_grp[shot_grp.is_interior, :]
    if nrow(interior) < 3
        continue
    end

    fd  = interior.V_dot_FD_s1em3
    nn  = interior.V_dot_NN
    rom = interior.V_dot_Romero

    stats_nn     = compute_stats(nn,  fd)
    stats_romero = compute_stats(rom, fd)

    push!(pershot_rows, (
        shot_id        = sid,
        split          = split_lbl,
        n_interior     = nrow(interior),
        R2_NN          = stats_nn.R2,
        Pearson_NN     = stats_nn.Pearson,
        RMSE_NN        = stats_nn.RMSE,
        R2_Romero      = stats_romero.R2,
        Pearson_Romero = stats_romero.Pearson,
        RMSE_Romero    = stats_romero.RMSE,
    ))
end

pershot_df = DataFrame(pershot_rows)
CSV.write(PERSHOT_OUT, pershot_df)
println("  Saved $(nrow(pershot_df)) per-shot rows → $PERSHOT_OUT")

# ============================================================================
# PER-POINT CSV
# ============================================================================
println("\nSaving per-point data (with V_dot_Romero baseline column) ...")
CSV.write(PERPOINT_OUT, df)
println("  Saved $(nrow(df)) rows × $(ncol(df)) cols → $PERPOINT_OUT")

# ============================================================================
# PATHOLOGICAL SHOTS
# ============================================================================
println("\nIdentifying pathological shots ...")

df_path_filter = df[df.is_interior, :]
fd_col_sym = Symbol("V_dot_FD_$HEADLINE_SMOOTHING")
mask_clipped = .!middle_fraction_mask(df_path_filter[!, fd_col_sym], CLIP_FRACTION)
df_clipped_points = df_path_filter[mask_clipped, :]

clipped_count_per_shot = combine(groupby(df_clipped_points, [:shot_id, :split]),
                                  nrow => :n_clipped_points)
sort!(clipped_count_per_shot, :n_clipped_points, rev=true)
pathological = clipped_count_per_shot[clipped_count_per_shot.n_clipped_points .> PATHOLOGICAL_THRESHOLD, :]

CSV.write(PATHOLOGICAL_OUT, pathological)
println("  $(nrow(pathological)) pathological shots (>$PATHOLOGICAL_THRESHOLD clipped points) → $PATHOLOGICAL_OUT")
if nrow(pathological) > 0 && nrow(pathological) <= 10
    println("  Pathological shots:")
    for r in eachrow(pathological)
        println("    shot $(r.shot_id) ($(r.split)): $(r.n_clipped_points) clipped points")
    end
elseif nrow(pathological) > 10
    println("  Top 10 pathological shots:")
    for r in eachrow(pathological[1:10, :])
        println("    shot $(r.shot_id) ($(r.split)): $(r.n_clipped_points) clipped points")
    end
end

# ============================================================================
# HEADLINE CONSOLE OUTPUT
# ============================================================================
println("\n" * "="^70)
println("=== V_dot DIAGNOSTIC HEADLINE — MlpODE_v70 ===")
println("="^70)

headline_row = filter(r -> r.split == "test"
                          && r.smoothing == HEADLINE_SMOOTHING
                          && r.clipping == "middle99"
                          && r.point_filter == "interior_only",
                      summary_df) |> first

println("\nTEST set, s=1e-3*N FD smoothing, middle 99% clip, interior points only")
println("  n = $(headline_row.n_points) points")
println()
@printf "  %-28s R² = %.4f, Pearson = %+.4f, Spearman = %+.4f, RMSE = %.3f\n" "Pure Romero physics:" headline_row.R2_Romero headline_row.Pearson_Romero headline_row.Spearman_Romero headline_row.RMSE_Romero
@printf "  %-28s R² = %.4f, Pearson = %+.4f, Spearman = %+.4f, RMSE = %.3f\n" "NN-augmented (MlpODE v70):" headline_row.R2_NN     headline_row.Pearson_NN     headline_row.Spearman_NN     headline_row.RMSE_NN
println()
@printf "  ΔR² (NN over Romero):       %+.4f\n" headline_row.R2_improvement_NN_over_Romero

println()
println("Per-shot R² distribution (TEST, NN, headline smoothing, interior):")
test_pershot = pershot_df[pershot_df.split .== "test", :]
if nrow(test_pershot) > 0
    @printf "  median R² = %.4f\n" median(test_pershot.R2_NN)
    @printf "  IQR R²    = [%.4f, %.4f]\n" quantile(test_pershot.R2_NN, 0.25) quantile(test_pershot.R2_NN, 0.75)
    @printf "  min, max  = [%.4f, %.4f]\n" minimum(test_pershot.R2_NN) maximum(test_pershot.R2_NN)
    @printf "  shots with R² > 0.5: %d / %d\n" sum(test_pershot.R2_NN .> 0.5) nrow(test_pershot)
end

println("\n=== Comparison reminder (other variants on test set, interior, middle-99%) ===")
println("  Pure Romero physics:  R² = 0.0002, Pearson = +0.0151")
println("  RomeroNN_v69:         R² = 0.0066, Pearson = -0.0811")
println("  MlpODE_v69:           R² = 0.0075, Pearson = +0.0865")
println("  RomeroNN_v70:         R² = 0.0141, Pearson = -0.1186")

println("\nAll outputs saved. Done.")