# ============================================================================
# preview_figure_grid.jl
#
# Draft of the 2×2 assignment figure.
# Layout:
#   (a) upper-left  — Li trajectory: data, Romero, RomeroNN_v69, MlpODE_v69
#   (b) upper-right — Ip trajectory: data, Romero, RomeroNN_v69, MlpODE_v69
#   (c) lower-left  — V  trajectory: data, Romero, RomeroNN_v69, MlpODE_v69
#   (d) lower-right — V_dot diagnostic scatter: RomeroNN_v69 vs FD,
#                                                MlpODE_v69 vs FD
#
# Color scheme (consistent across all panels):
#   data            → black solid
#   Pure Romero     → gray dashed (de-emphasized baseline)
#   RomeroNN_v69    → :steelblue
#   MlpODE_v69      → :darkorange
#
# Inputs:
#   predictions_RomeroNN_v69.csv      (single-shot trajectory)
#   predictions_MlpODE_v69.csv        (same shot, different model)
#   vdot_diagnostic_perpoint_RomeroNN_v69.csv
#   vdot_diagnostic_perpoint_MlpODE_v69.csv
#
# Output:
#   figure_grid_preview_v69.png
# ============================================================================

using CSV
using DataFrames
using Statistics
using Plots
using Printf
using LaTeXStrings
using FileIO

# ============================================================================
# CONFIG
# ============================================================================
const COLOR_DATA      = :black
const COLOR_ROMERO    = :gray
const COLOR_ROMERONN  = :steelblue
const COLOR_MLPODE    = :darkorange

const LW_DATA   = 2.0
const LW_ROMERO = 1.5
const LW_NN     = 1.8

const SHOT_LABEL = "shot 1050218005 (test set)"

default(legendfontsize = 10)
# ============================================================================
# UTILITY
# ============================================================================
function middle_fraction_mask(reference, keep_fraction)
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
# LOAD DATA
# ============================================================================
println("Loading single-shot trajectory CSVs ...")
df_traj_romeroNN = CSV.read("predictions_RomeroNN_v68.csv", DataFrame)
df_traj_mlpODE   = CSV.read("predictions_MlpODE_v69.csv",   DataFrame)
println("  RomeroNN trajectory: $(nrow(df_traj_romeroNN)) timesteps")
println("  MlpODE   trajectory: $(nrow(df_traj_mlpODE)) timesteps")

# Use RomeroNN's file as the source of truth for data and Romero baseline.
# (Both files should have identical data and Romero columns since they're
# computed from the same shot.)
t_arr   = df_traj_romeroNN.time
li_data = df_traj_romeroNN.li_data
ip_data = df_traj_romeroNN.ip_data
V_data  = df_traj_romeroNN.V_data
li_rom  = df_traj_romeroNN.li_romero
ip_rom  = df_traj_romeroNN.ip_romero
V_rom   = df_traj_romeroNN.V_romero

li_rNN  = df_traj_romeroNN.li_nn
ip_rNN  = df_traj_romeroNN.ip_nn
V_rNN   = df_traj_romeroNN.V_nn

li_mlp  = df_traj_mlpODE.li_nn
ip_mlp  = df_traj_mlpODE.ip_nn
V_mlp   = df_traj_mlpODE.V_nn

println("Loading diagnostic per-point CSVs ...")
df_diag_romeroNN = CSV.read("vdot_diagnostic_perpoint_RomeroNN_v69.csv", DataFrame)
df_diag_mlpODE   = CSV.read("vdot_diagnostic_perpoint_MlpODE_v69.csv",   DataFrame)

function prep_diag(df)
    df_t = df[(df.split .== "test") .& df.is_interior, :]
    mask = middle_fraction_mask(df_t.V_dot_FD_s1em3, 0.99)
    return df_t[mask, :]
end

df_diag_rNN_clip = prep_diag(df_diag_romeroNN)
df_diag_mlp_clip = prep_diag(df_diag_mlpODE)

r_rNN = cor(df_diag_rNN_clip.V_dot_NN, df_diag_rNN_clip.V_dot_FD_s1em3)
r_mlp = cor(df_diag_mlp_clip.V_dot_NN, df_diag_mlp_clip.V_dot_FD_s1em3)

# ============================================================================
# PANEL (a) — Li
# ============================================================================
println("\nBuilding panel (a) Li ...")
plt_a = plot(t_arr, li_data;
    label = L"data",
    color = COLOR_DATA,
    linewidth = LW_DATA,
    xlabel = L"\textbf{Time}\ \mathrm{(s)}",
    ylabel = L"L_i\ \textbf{(dimensionless)}",
    title = L"\textbf{(a)\ Internal\ Inductance\ } L_i",
    titlefontsize = 14,
    titleloc = :center,
    legend = :topleft,
    grid = true,
    size = (600, 500),
)
plot!(plt_a, t_arr, li_rom;
    label = L"Romero",
    color = COLOR_ROMERO,
    linestyle = :dash,
    linewidth = LW_ROMERO,
)
plot!(plt_a, t_arr, li_rNN;
    label = L"RomeroNN",
    color = COLOR_ROMERONN,
    linewidth = LW_NN,
)
plot!(plt_a, t_arr, li_mlp;
    label = L"MlpODE",
    color = COLOR_MLPODE,
    linewidth = LW_NN,
)

# ============================================================================
# PANEL (b) — Ip
# ============================================================================
println("Building panel (b) Ip ...")
plt_b = plot(t_arr, ip_data;
    label = "data",
    color = COLOR_DATA,
    linewidth = LW_DATA,
    xlabel = L"\textbf{Time}\ \mathrm{(s)}",
    ylabel = L"I_p\ \textbf{(MA)}",
    title = L"\textbf{(b)\ Plasma\ Current\ } I_p",
    titlefontsize = 14,
    titleloc = :center,
    legend = false,         # legend already in (a)
    grid = true,
)
plot!(plt_b, t_arr, ip_rom;  color = COLOR_ROMERO, linestyle = :dash, linewidth = LW_ROMERO, label = "")
plot!(plt_b, t_arr, ip_rNN;  color = COLOR_ROMERONN, linewidth = LW_NN, label = "")
plot!(plt_b, t_arr, ip_mlp;  color = COLOR_MLPODE,   linewidth = LW_NN, label = "")

# ============================================================================
# PANEL (c) — V
# ============================================================================
println("Building panel (c) V ...")
plt_c = plot(t_arr, V_data;
    label = "data",
    color = COLOR_DATA,
    linewidth = LW_DATA,
    xlabel = L"\textbf{Time}\ \mathrm{(s)}",
    ylabel = L"V\ \textbf{(V)}",
    title = L"\textbf{(c)\ Loop\ Voltage\ } V",
    titlefontsize = 14,
    titleloc = :center,
    legend = false,
    grid = true,
)
plot!(plt_c, t_arr, V_rom;  color = COLOR_ROMERO, linestyle = :dash, linewidth = LW_ROMERO, label = "")
plot!(plt_c, t_arr, V_rNN;  color = COLOR_ROMERONN, linewidth = LW_NN, label = "")
plot!(plt_c, t_arr, V_mlp;  color = COLOR_MLPODE,   linewidth = LW_NN, label = "")

# ============================================================================
# PANEL (d) — V_dot diagnostic scatter
# ============================================================================
println("Building panel (d) V_dot scatter ...")

all_y = vcat(df_diag_rNN_clip.V_dot_NN, df_diag_mlp_clip.V_dot_NN)
all_x = df_diag_rNN_clip.V_dot_FD_s1em3
lo = min(minimum(all_x), minimum(all_y))
hi = max(maximum(all_x), maximum(all_y))
pad = 0.05 * (hi - lo)
axis_lo = lo - pad
axis_hi = hi + pad

plt_d = scatter(
    df_diag_rNN_clip.V_dot_FD_s1em3, df_diag_rNN_clip.V_dot_NN;
    label = L"\mathrm{RomeroNN}\ (r = -0.081,\ R^2 = 0.0066)",
    markersize = 3,
    markershape = :utriangle,
    markeralpha = 0.45,
    markerstrokewidth = 0,
    color = COLOR_ROMERONN,
    xlabel = L"\frac{dV}{dt} \textbf{\ from\ data}",
    ylabel = L"\frac{dV}{dt} \textbf{\ from\ NN}",
    title  = L"\textbf{(d)\ } \frac{dV}{dt} \textbf{\ Correlation:\ Data\ vs\ NN\ Predictions}",
    titlefontsize = 14,
    titleloc = :center,
    xlims = (-30, 30),
    ylims = (-30, 30),
    aspect_ratio = :equal,
    legend = :bottomright,
    grid = true,
)
scatter!(plt_d,
    df_diag_mlp_clip.V_dot_FD_s1em3, df_diag_mlp_clip.V_dot_NN;
    label = L"\mathrm{MlpODE}\ (r = +0.086,\ R^2 = 0.0075)",
    markersize = 3.5,
    markershape = :circle,
    markeralpha = 0.45,
    markerstrokewidth = 0,
    color = COLOR_MLPODE,
)
plot!(plt_d, [-30, 30], [-30, 30];
    label = L"\mathrm{Ideal:\ NN = Data}",
    color = :black,
    linestyle = :dash,
    linewidth = 1.5,
)

annotate!(plt_d, 30 * 0.6, 30 * 0.7,
    text("perfect agreement", 9, :gray, rotation = 45))

# ============================================================================
# COMPOSE 2x2 GRID
# ============================================================================
println("\nComposing 2×2 grid ...")

grid = plot(plt_a, plt_b, plt_c, plt_d;
    layout = (2, 2),
    size = (1100, 950),
    plot_title = L"\textbf{Trained\ Neural\ ODEs\ Track\ } \mathbf{L_i} \textbf{\ and\ } \mathbf{I_p} \textbf{\ but\ Not\ } \mathbf{V} \textbf{\ —\ shot\ 1050218005}",
    plot_titlefontsize = 18,
    plot_titlefontweight = :bold,
    margin = 5Plots.mm,
)

savefig(grid, "figure_grid_preview_v69_v19.png")
println("Saved → figure_grid_preview_v69_v19.png")
run(`sips -s format jpeg figure_grid_preview_v69_v19.png --out figure_grid_preview_v69_v19.jpg`)
println("Saved both PNG and JPEG.")
println("Done.")