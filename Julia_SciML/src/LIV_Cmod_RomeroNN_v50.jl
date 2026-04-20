using ComponentArrays
using Lux 
using Glob
using DiffEqFlux
using OrdinaryDiffEq
using SciMLSensitivity
using Optimization
using OptimizationOptimJL
using OptimizationOptimisers
using OptimizationPolyalgorithms
using Random
using Plots
using DataFrames
using CSV
using Dierckx
using Flux: destructure

#medium.com/@sliencepill/easily-read-multiple-csv-file....
path_to_csv_dir = "/Users/cmason83/Desktop/sciml_workspaces/wang_model/wang_model_workspace/src/romero_shots_489"
csv_files = glob("*.csv", path_to_csv_dir)
df_list = DataFrame.(CSV.File.(csv_files)) #creates a list of dataframe all with identical headers
df_489_shots = reduce(vcat, df_list)
#sort!(df_489_shots.time, alg=QuickSort )
#kappa - free parameter
#tau - free parameter
#ip_MA- plamsa current
#li - plasma internal inductance
#Vind - control variable
#vc_minus_vb - just...V 

function copy_headers_excluding(df::DataFrame, column_to_exclude::Symbol)
   all_headers = names(df)
   print("all 11 headers are: ")
   println(all_headers)
   filtered_headers = filter(header -> header != column_to_exclude, all_headers)
   return filtered_headers
end

function time_differences(time_array)
    println("we are in the function count_positive_deltat. I think we are going to hit an error")
    L = length(time_array)
    time_diff = Array{Float64}(undef, L)
    positive_diff = 0
    diff2 = 0.01999998 #arbitrary
    time_diff[1] = 0.001
    for idx ∈ 2:L 
        diff = time_array[idx] - time_array[idx-1]
        println("diff is $diff at index $idx")
        if (diff <= 0)
            diff = time_diff[1]
        end
        
        time_diff[idx] = diff
        if diff >= 0
            positive_diff = positive_diff + 1
        end
    end

    println("$(positive_diff) items out of $(L) are positive")
    println("returning time_diff")
    return time_diff
end

function cumulative_time(time_array)
    L = length(time_array)
    time_diff = time_differences(time_array)
  
    print("do we get here?")
    L_time_diff = length(time_diff)
    print("length of time_diff array is $L_time_diff\n")
    c_time = Array{Float64}(undef, L)
    c_time[1] = 0.
    
    time_diff_L = length(time_diff)
    println("time_diff_L is $time_diff_L")
    #print("time_diff is $(time_diff)")
    for (idx, δ) ∈ enumerate(time_diff)
  
      println("index is $idx and δ is $δ")
      if idx > 1
         c_time[idx] = c_time[idx - 1] + δ
         println("c_time $idx is now $(c_time[idx])")
      end
    end
    return c_time
  end

function check_inputs(times, Vinds, dVinds)
   @assert times isa AbstractVector "time array (c_time_array_20) must be a vector"
   @assert Vinds isa AbstractVector "value array (Vind_array_20) must be a vector"
   @assert dVinds isa AbstractVector "derivative arrays (dVind_array_20) must be a vector"

   @assert eltype(times) <: Real "time array must contain numbers"
   @assert eltype(Vinds) <: Real "Vinds array must contain numbers"
   @assert eltype(dVinds) <: Real "derivative dVinds array must contains numbers"

   @assert length(times) == length(Vinds) == length(dVinds) "c_times_array_20, Vind_array_20, and dVind_array_20 must have equal length"
   @assert all(diff(times) .> 0) "time array must be strictly increasing"

   println("Inputs look good: length = $(length(times)) element types ($(eltype(times)), $(eltype(Vinds)), $(eltype(dVinds))).")
end
#ok let's apply a cubic hermite spline to Vind based on the guidance from Raj's document:
#'Building a Causal Cubic Hermite Spline in Julia'
#=
Assume: 
time_data :: Vector{Float32} #e.g. 0.0, 0.01, 0.02, .... , 98.70
Vind_data :: Vector{Float32} # control values Vind at each t_data
=#
# ---------DATA PREPARATION v24 -----------

const T = Float32

# ── SINGLE SHOT EXTRACTION ──────────────────────────────────────────
#shot_id   = 1050218005 # v < 32
#shot_id = 1140520018   # v32
shot_id = 1050218005    # v42
one_shot  = df_489_shots[df_489_shots.shot .== shot_id, :]

# relative time starting from 0
t_shot    = T.(one_shot.time .- one_shot.time[1])
li_shot   = T.(one_shot.li)
ip_shot   = T.(one_shot.ip_MA)
V_shot    = T.(one_shot.vc_minus_vb)
Vind_shot = T.(one_shot.Vind)

@assert all(diff(t_shot) .> 0) "time must be strictly increasing"
@show length(t_shot)       # should be 28
@show t_shot[end]          # should be ~0.54 seconds

# ── VIND SPLINE FOR THIS SHOT ────────────────────────────────────────
N          = length(t_shot)
dVind      = Vector{T}(undef, N)
dVind[1]   = (Vind_shot[2] - Vind_shot[1]) / (t_shot[2] - t_shot[1])
for i in 2:N
    dVind[i] = (Vind_shot[i] - Vind_shot[i-1]) / (t_shot[i] - t_shot[i-1])
end

smoothing     = 1e-3 * N
spl_shot      = Spline1D(t_shot, Vind_shot; k=3, bc="extrapolate", s=smoothing)
Vind_spline32(t) = T(spl_shot(t))

# ── PHYSICS CONSTANTS ────────────────────────────────────────────────
max_ip_MA  = maximum(abs.(T.(df_489_shots.ip_MA)))
max_li     = maximum(abs.(T.(df_489_shots.li)))
liip_norm  = T(max_ip_MA * max_li)
κ_empirical = T(0.98)
τ           = T(1.25)

u₀ = T[li_shot[1], ip_shot[1], V_shot[1]]
params_romero32 = (k = κ_empirical, τ = τ, liip_norm = liip_norm)

@show u₀
@show liip_norm

# --------END DATA PREPARATION v24 ---------

@show minimum(abs.(li_shot)), minimum(abs.(ip_shot))

eps = 1e-9
function romero_ODEfunc!(du::AbstractVector{T}, u::AbstractVector{T}, p, t::T)

    Li, Ip, V  = u       
    ## Li - plasma internal inductance
    #Ip - plasma current
    #V - "a summary statistic of spatial variables"
    #Vind - control variable
    Vind = Vind_spline32(t)
    
    #Li denomimator
    dli_dt_denom = p.liip_norm * (abs(Ip) + eps)
    dip_dt_denom = p.liip_norm * (abs(Li) + eps) #avoids 1/0 error?

    #du[1] is dLi/dt 
    du[1] =  (-2 * Vind - 2 * V)/(dli_dt_denom)   #(-2*V_ind - 2 * V)*(1/Ip) 
    #du[2] is dIp/dt
    du[2] = (2 * Vind + V)/(dip_dt_denom)
    #du[3] is dV/dt 
    du[3] = -V/p.τ + (p.k/p.τ)* Vind #<---- in RomeroNNV, only the RHS of this equation is replace by a NN
    return nothing 
    #In MlpODE, we have three neural networks for the RHS of dL/dt, dI/dt, and dV/dt respecitvely
end

#c_time_length = length(c_time_array)
#tspan = (first(c_time_array), last(c_time_array))
#tsteps = range(tspan[1], tspan[2], length = c_time_length)
#saveat = c_time_array

# Your data time grid (must be strictly increasing Float64)
tspan32 = (t_shot[1], t_shot[end])
saveat = t_shot

f_romero32 = ODEFunction(romero_ODEfunc!)    #ensures in-place form
romero_trueode = ODEProblem(f_romero32, u₀, tspan32, params_romero32)

#sanity: confirm in-place
@show romero_trueode.f isa SciMLBase.ODEFunction{true}
#@show eltype(romero_trueode.u₀), typeof(romero_trueode.p), tspan32
@show eltype(romero_trueode.u0), typeof(romero_trueode.p), tspan32 

#Generating the ground truth data
using SciMLBase

@show typeof(romero_trueode)

@show typeof(romero_trueode.p) romero_trueode.p
@show romero_trueode.f isa SciMLBase.ODEFunction
@show romero_trueode.f.mass_matrix # sanity, often `nothing`
@show romero_trueode.p isa SciMLBase.NullParameters #should be false


# Check V statistics across all shots
using Statistics
println("V_data mean: ", mean(df_489_shots.vc_minus_vb))
println("V_data std: ", std(df_489_shots.vc_minus_vb))
println("V_data min: ", minimum(df_489_shots.vc_minus_vb))
println("V_data max: ", maximum(df_489_shots.vc_minus_vb))

# Check shot 1050218005 specifically
shot_V = df_489_shots[df_489_shots.shot .== 1050218005, :vc_minus_vb]
println("\nShot 1050218005 V values: ", shot_V)

#added for v31
# Check shot 1050218005 specifically
shot_V = df_489_shots[df_489_shots.shot .== 1050218005, :vc_minus_vb]
println("\nShot 1050218005 V values: ", shot_V)

# Find shots with gentlest V dynamics
shot_V_stats = combine(groupby(df_489_shots, :shot), 
    :vc_minus_vb => std => :V_std,
    :vc_minus_vb => maximum => :V_max,
    :vc_minus_vb => minimum => :V_min,
    nrow => :count)

sort!(shot_V_stats, :V_std)
println("10 shots with gentlest V dynamics:")
println(shot_V_stats[1:10, :])

#-- end of added for v31 
using Zygote
ZYG = InterpolatingAdjoint(autojacvec = ZygoteVJP())

# 2) tell Zygote not to differentiate through the spline (we don't need grads w.r.t. spline coeffs)
Zygote.@nograd Vind_spline32

true_romero_data = Array(solve(romero_trueode, Tsit5(); saveat = t_shot, reltol=1e-6, abstol=1e-8, verbose=false))
#Universal ODE: Part 1 RomeroNNV
#Random.seed!(1234)
Random.seed!(42)
rng1 = Random.default_rng()

#Set up the RomeroNNV neural network
#input: [ip, li, vc_minus_vb, Vind] ---> output V

# Build the Lux model
romero_NN = Lux.Chain(
    Lux.Dense(4, 32, softplus),
    Lux.Dense(32, 32, softplus),
    Lux.Dense(32, 1, tanh) # tanh bounds output to (-1 to +1)
)

# Initialize Lux params/state (both NamedTuples)
p_nn0, st_nn = Lux.setup(rng1, romero_NN)
Zygote.@nograd st_nn

# Flatten params -> θ::Vector and a reconstructor re(θ) -> NamedTuple params
θ0, re = destructure(p_nn0)
θ0 = T.(θ0)

# Near-zero initialization of last layer so NN correction starts small
# Final Dense(32,1) has 32 weights + 1 bias = 33 params at the end of θ0
θ0[end-32:end] .= T(0.01) .* θ0[end-32:end]  #v34

#using StaticArrays

softabs(x, eps=T(1e-9)) = sqrt(x*x + eps*eps)
function dudt_romeroNN(du, u, θ, t)
    Li, Ip, V = u
    #Vind = Vind_spline(t)
    Vind = Vind_spline32(t)

    # Romero physics parts (use your liip_norm etc.)
    du[1] = (-2*Vind - 2*V) / (liip_norm * softabs(Ip))
    du[2] = ( 2*Vind +    V) / (liip_norm * softabs(Li))

    # Reconstruct Lux params from θ and evaluate the NN
    ps = re(θ)                                   # NamedTuple of params
    x  = reshape(T[Li, Ip, V, Vind], 4, 1)  # 4×1 matrix, batch size 1           # small static vector is fastest
    y, _ = Lux.apply(romero_NN, x, ps, st_nn)    # st_nn is the (empty) state for Dense
    
     # Physics baseline + learned residual (this is the actual RomeroNNV formulation)
     du[3] = -V/τ + (κ_empirical/τ)*Vind + y[1] # added in v33
end

# ODE problem for the neural surrogate
prob_pred_romeroNN = ODEProblem(dudt_romeroNN, u₀, tspan32, θ0)

# ── TRAINING DATA FOR SINGLE SHOT v24 ───────────────────────────────────
training_tsteps   = t_shot
training_datasize = length(training_tsteps)

training_data = (
    t      = t_shot,
    y      = (li = li_shot, ip = ip_shot, V = V_shot),
    Vind   = Vind_spline32,
    liip_norm = liip_norm
)

ode_data_ms = permutedims(hcat(li_shot, ip_shot, V_shot))  # 3 × 28

@show size(ode_data_ms)        # should be (3, 28)
@show length(training_tsteps)  # should be 28

#  ==== END TRAINNG BLOCK v24 -------
# after you define `training_data`
# Use Li and Ip as supervised targets (2×T). Add V as a 3rd row only if you want to supervise it.
#ode_data_ms = reduce(hcat, (training_data.y.li, training_data.y.ip, training_data.y.V))
# => size(ode_data_ms) == (2, length(training_tsteps))

# Use Li and Ip as supervised targets: 2 × T
#ode_data_ms = permutedims(hcat(training_data.y.li, training_data.y.ip))  # 2 × T
# If you also want to supervise V, include it too:
ode_data_ms = permutedims(hcat(training_data.y.li, training_data.y.ip, training_data.y.V))  # 3 × T
ode_data_ms = T.(ode_data_ms)

@show size(ode_data_ms)                  # should be (n_obs, T)
@show length(training_tsteps)            # should equal T

@assert size(ode_data_ms, 2) == length(training_tsteps) "columns of ode_data_ms must match length(tsteps)"

Tpts = length(training_tsteps)              # <-- not size(ode_data_ms, 2) is also fine, but be consistent
group_size = min(5, Tpts)                   # pick what you want, but cap to T
@assert 2 ≤ group_size ≤ Tpts "group_size must be in 2…$Tpts"

# Non-overlapping chunking for plotting multi-shoot segments (simple helper)
group_ranges(n, g) = [i:min(i+g-1, n) for i in 1:g:n]

## LOSS AND CALLBACK FUNCTIONS FROM HARE-LYNX-example
function plot_multiple_shoot(plt, preds, group_size)
    step = group_size - 1
    ranges = group_ranges(training_datasize, group_size)
    for (i, rg) in enumerate(ranges)
        plot!(plt, training_tsteps[rg], preds[i][1,:], markershape=:circle, label="Group $(i)", legend=false)
    end
end

##input losses from the traing for visualizations
losses = []
#it is possible that the optimal is met and crosses over and is accessible by pred_list
pred_list = []
#option to save animaton gif later on
anim = Plots.Animation()

iter = 0
callback = function(p, l; doplot = false)
    #display(l)
    push!(losses, l)
    #push!(pred_list, preds)
    global iter
    iter += 1
    if iter % 100 == 0
        println("Iter $iter: loss = $l")
    end
    if doplot
        #plot the original data - what does this mean?
        #are we trying to plot....ip_MA? li? vc_minus_vb?
        plt = scatter(training_tsteps, training_data.y.ip; label = "ip_MA", legend=false)
        #plot the differential predictions for individual shoots
        plot_multiple_shoot(plt, preds, group_size)
        #needed for gif of plots of networks and data
        frame(anim)

       if iter % 100 == 0
        display(plot(plt))
       end
    end

    return false
end 

#group size for each neural network and the penalty for continuity
#The larger the group size, the smaller the continuity_term
#The smaller the group_size, the larger the continuity_term

#OK... I do not understand this
#According to ADavis89, the group_size was 15 and the continuity term was 14
#According Steve Frank's paper - which applied multishooting to the hare lynx dataset - the group_size was 
#the number of variables being explicitly modeled (i.e. lynx and hare populations) plus the  maxiumum number of hidden 
#latent terms in the differential equation

#I do not understand the role of the continuity term at this point 

#choosing these values from Allen Wang's paper
# Use Li and Ip as supervised targets (2 × T); add V only if you really want to supervise it too.
#ode_data_ms = [training_data.y.li  training_data.y.ip];   # 2 × T

# If you want to include V as well: 
# ode_data_ms = [training_data.y.li  training_data.y.ip  training_data.y.V]';  # 3 × T


#group_size = 4
continuity_term = 0

#predict adjoint for romeroNNV
function predict_adjoint_romeroNN(θ, t)
    x = Array(
            solve(prob_pred_romeroNN,
            Tsit5(), p = θ,
            saveat = T.(t),
            sensealg = ZYG, #InterpolatingAdjoint(autojacvec = ReverseDiffVJP(true))))
            reltol=1e-6,
            abstol=1e-8,
            verbose=false
            ))
end

#Get back to this.... how do I call the Huber loss function with just the Li and Ip data?
#Find an example on the sciml julia documentation
#=
function loss_adjoint_huber(θ)
    x = predict_adjoint_romeroNN(θ, training_tsteps)
    #compare with the actual data from cmod
    #need data for.... just I and L?
    return Flux.huber_loss(x, training_data, δ = 0.01)
    
end 
=#

function ms_huber_loss(pred, data; δ = 0.1)
    eps = 1e-9
    @views begin
        r1 = abs.(data[1, :] .- pred[1, :]) ./ (abs.(data[1, :]) .+ eps)
        r2 = abs.(data[2, :] .- pred[2, :]) ./ (abs.(data[2, :]) .+ eps)
    end
    hub(x) = x ≤ δ ? 0.5*x^2/δ : x - 0.5*δ
    vals = map(hub, r1) .+ map(hub, r2)
    return sum(vals) / length(vals)
end

#according to google this a what the Huber loss function actually does...
#=
function my_huber_loss(ŷ, y, δ = 0.01, agg = mean)
    diff = abs.(ŷ .- y)
    loss = zeros(size(diff))
    diff_len = length(diff)
    for i in 1:diff_len
        if diff[i] < δ
            loss[i] = 0.5 * diff[i]^2
        else
            loss[i] = δ * (diff[i] - 0.5 * δ)
        end
    end
    agg(loss)
end

function basic_loss_function(data, pred)
    return sum(abs2, data - pred)
end
=#


# ── MULTI-SHOT TRAINING v38 ──────────────────────────────────────────
N_shots = 100 # v41
shot_ids = unique(df_489_shots.shot)[1:N_shots]


# Precompute all shot splines BEFORE the loss function v40 
shot_splines = Dict{Int64, Spline1D}()
for sid in shot_ids
    shot = df_489_shots[df_489_shots.shot .== sid, :]
    t_s = T.(shot.time .- shot.time[1])
    Vind_s = T.(shot.Vind)
    shot_splines[sid] = Spline1D(t_s, Vind_s; k=3, bc="extrapolate",
                                  s=Float64(1e-3)*length(t_s))
end
# Precompute all shot data BEFORE the loss function v41
shot_data = Dict{Int64, NamedTuple}()
for sid in shot_ids
    shot = df_489_shots[df_489_shots.shot .== sid, :]
    t_s = T.(shot.time .- shot.time[1])
    shot_data[sid] = (
        t_s  = t_s,
        u0_s = T[shot.li[1], shot.ip_MA[1], shot.vc_minus_vb[1]],
        li   = T.(shot.li),
        ip_MA   = T.(shot.ip_MA)
    )
end

function loss_multishot(θ)
    total_loss = 0.0f0
    n_valid = 0
    for sid in shot_ids
        d = shot_data[sid]
        t_s = d.t_s
        u0_s = d.u0_s
        spl_s = shot_splines[sid]
        function dudt_shot(du, u, θ, t)
            Li, Ip, V = u
            Vind = T(Zygote.ignore(() -> spl_s(t))) #changded for v39
            du[1] = (-2*Vind - 2*V) / (liip_norm * softabs(Ip))
            du[2] = ( 2*Vind +    V) / (liip_norm * softabs(Li))
            ps = re(θ)
            x = reshape(T[Li, Ip, V, Vind], 4, 1)
            y, _ = Lux.apply(romero_NN, x, ps, st_nn)
            du[3] = -V/τ + (κ_empirical/τ)*Vind + y[1]
        end
        prob_s = ODEProblem(dudt_shot, u0_s, (t_s[1], t_s[end]), θ)
        sol = solve(prob_s, Tsit5(), p=θ, saveat=t_s,
                    sensealg=ZYG, reltol=1f-4, abstol=1f-6,
                    verbose=false)
        if sol.retcode == ReturnCode.Success
            pred = Array(sol)
            li_data = d.li
            ip_data = d.ip_MA
            eps_l = T(1e-9)
            r1 = abs.(li_data .- pred[1,:]) ./ (abs.(li_data) .+ eps_l)
            r2 = abs.(ip_data .- pred[2,:]) ./ (abs.(ip_data) .+ eps_l)
            hub(x) = x ≤ T(0.1) ? T(0.5)*x^2/T(0.1) : x - T(0.05)
            total_loss += sum(map(hub, r1) .+ map(hub, r2)) / length(t_s)
            n_valid += 1
        end
    end
    return n_valid > 0 ? total_loss / n_valid : T(1000.0)
end

adtype  = Optimization.AutoZygote()
optf    = Optimization.OptimizationFunction((x, p) -> loss_multishot(x), adtype)
optprob = Optimization.OptimizationProblem(optf, θ0)

#v42 - exponential decay learning rate via AdamW
res_ms  = Optimization.solve(optprob,
    OptimizationOptimisers.AdamW(0.001f0, (0.9f0, 0.999f0), 1f-4);
    callback = callback,
    maxiters = 3000)

# ── DIAGNOSTIC v28: three-way comparison ────────────────────────────
θ_trained = res_ms.minimizer

# Check how many shots solved successfully on final pass - v43
n_success = 0
n_fail = 0
for sid in shot_ids
    d = shot_data[sid]
    spl_s = shot_splines[sid]
    function dudt_check(du, u, θ, t)
        Li, Ip, V = u
        Vind = T(Zygote.ignore(() -> spl_s(t)))
        du[1] = (-2*Vind - 2*V) / (liip_norm * softabs(Ip))
        du[2] = ( 2*Vind +    V) / (liip_norm * softabs(Li))
        ps = re(θ_trained)
        x = reshape(T[Li, Ip, V, Vind], 4, 1)
        y, _ = Lux.apply(romero_NN, x, ps, st_nn)
        du[3] = -V/τ + (κ_empirical/τ)*Vind + y[1]
    end
    prob_c = ODEProblem(dudt_check, d.u0_s, (d.t_s[1], d.t_s[end]), θ_trained)
    sol_c = solve(prob_c, Tsit5(), p=θ_trained, saveat=d.t_s,
                  reltol=1f-4, abstol=1f-6, verbose=false)
    sol_c.retcode == ReturnCode.Success ? ( global n_success += 1) : ( global n_fail += 1)
end
println("Shots solved successfully: $n_success / $(n_success + n_fail)")

# Compute per-shot percent errors for Romero vs NN --- added for v46
romero_li_errors = Float32[]
romero_ip_errors = Float32[]
nn_li_errors = Float32[]
nn_ip_errors = Float32[]

for sid in shot_ids
    d = shot_data[sid]
    spl_s = shot_splines[sid]
    
    # Pure Romero solve
    
    # Float64 Romero solve for stable MAPE evaluation
    # Float64 Romero function for stable MAPE evaluation
    function romero_ODEfunc_f64!(du, u, p, t)
        Li, Ip, V = u
        Vind = spl_s(Float64(t))  # use the shot-specific spline, evaluated in Float64
        eps64 = 1e-9
        dli_denom = p.liip_norm * (abs(Ip) + eps64)
        dip_denom = p.liip_norm * (abs(Li) + eps64)
        du[1] = (-2*Vind - 2*V) / dli_denom
        du[2] = ( 2*Vind +   V) / dip_denom
        du[3] = -V/p.τ + (p.k/p.τ)*Vind
    end
    
    prob_r64 = ODEProblem(romero_ODEfunc_f64!,
                      Float64.(d.u0_s),
                      (Float64(d.t_s[1]), Float64(d.t_s[end])),
                      (k=Float64(κ_empirical), τ=Float64(τ),
                       liip_norm=Float64(liip_norm)))

    sol_r = solve(prob_r64, Tsit5(); saveat=Float64.(d.t_s), reltol=1e-6, abstol=1e-8, verbose=false)
    
    # NN solve
    function dudt_eval(du, u, θ, t)
        Li, Ip, V = u
        Vind = T(Zygote.ignore(() -> spl_s(t)))
        du[1] = (-2*Vind - 2*V) / (liip_norm * softabs(Ip))
        du[2] = ( 2*Vind +    V) / (liip_norm * softabs(Li))
        ps = re(θ_trained)
        x = reshape(T[Li, Ip, V, Vind], 4, 1)
        y, _ = Lux.apply(romero_NN, x, ps, st_nn)
        du[3] = -V/τ + (κ_empirical/τ)*Vind + y[1]
    end
    prob_n = ODEProblem(dudt_eval, d.u0_s, (d.t_s[1], d.t_s[end]), θ_trained)
    sol_n = solve(prob_n, Tsit5(); p=θ_trained, saveat=d.t_s, reltol=1f-4, abstol=1f-6)
    
    if sol_r.retcode == ReturnCode.Success && sol_n.retcode == ReturnCode.Success
        r_pred = Array(sol_r)
        n_pred = Array(sol_n)
        # Mean absolute percent error on Li and Ip
        push!(romero_li_errors, mean(abs.(d.li .- r_pred[1,:]) ./ abs.(d.li)))
        push!(romero_ip_errors, mean(abs.(d.ip_MA .- r_pred[2,:]) ./ abs.(d.ip_MA)))
        push!(nn_li_errors,     mean(abs.(d.li .- n_pred[1,:]) ./ abs.(d.li)))
        push!(nn_ip_errors,     mean(abs.(d.ip_MA .- n_pred[2,:]) ./ abs.(d.ip_MA)))
    end
end

println("=== Per-shot MAPE across $(length(nn_li_errors)) shots ===")
println("Romero Li MAPE:  $(round(100*mean(romero_li_errors), digits=3))%")
println("Romero Ip MAPE:  $(round(100*mean(romero_ip_errors), digits=3))%")
println("NN     Li MAPE:  $(round(100*mean(nn_li_errors),     digits=3))%")
println("NN     Ip MAPE:  $(round(100*mean(nn_ip_errors),     digits=3))%")
println("Shots excluded (Romero unstable): $(N_shots - length(nn_li_errors))")
# Pure Romero physics (no NN)
romero_pred = Array(solve(romero_trueode, Tsit5(); 
                          saveat = t_shot, 
                          reltol=1e-6, abstol=1e-8))

# NN prediction (forward pass)
pred_full = Array(solve(prob_pred_romeroNN, Tsit5(), 
                        p = θ_trained,
                        saveat = training_tsteps,
                        reltol = 1e-4,
                        abstol = 1e-6,
                        verbose = false))

@show size(pred_full)
@show length(training_tsteps)

n_solved = size(pred_full, 2)

using CSV, DataFrames
results_df = DataFrame(
    time         = training_tsteps[1:n_solved],
    li_data      = training_data.y.li[1:n_solved],
    li_romero    = romero_pred[1, 1:n_solved],
    li_nn        = pred_full[1, :],
    ip_data      = training_data.y.ip[1:n_solved],
    ip_romero    = romero_pred[2, 1:n_solved],
    ip_nn        = pred_full[2, :],
    V_data       = training_data.y.V[1:n_solved],
    V_romero     = romero_pred[3, 1:n_solved],
    V_nn         = pred_full[3, :]
)

CSV.write("predictions_v50.csv", results_df)
println("Saved predictions to predictions_v50.csv")

# v43 Evaluate on a hard shot from the training set
hard_shot_id = 1050218005
d_hard = shot_data[hard_shot_id]
spl_hard = shot_splines[hard_shot_id]

function dudt_hard(du, u, θ, t)
    Li, Ip, V = u
    Vind = T(Zygote.ignore(() -> spl_hard(t)))
    du[1] = (-2*Vind - 2*V) / (liip_norm * softabs(Ip))
    du[2] = ( 2*Vind +    V) / (liip_norm * softabs(Li))
    ps = re(θ_trained)
    x = reshape(T[Li, Ip, V, Vind], 4, 1)
    y, _ = Lux.apply(romero_NN, x, ps, st_nn)
    du[3] = -V/τ + (κ_empirical/τ)*Vind + y[1]
end

prob_hard = ODEProblem(dudt_hard, d_hard.u0_s, 
                       (d_hard.t_s[1], d_hard.t_s[end]), θ_trained)
sol_hard = solve(prob_hard, Tsit5(), p=θ_trained,
                 saveat=d_hard.t_s, reltol=1f-4, abstol=1f-6)

hard_df = DataFrame(
    time     = d_hard.t_s,
    li_data  = d_hard.li,
    li_nn    = Array(sol_hard)[1,:],
    ip_data  = d_hard.ip_MA,
    ip_nn    = Array(sol_hard)[2,:]
)
CSV.write("predictions_hard_shot_v50.csv", hard_df)
println("Saved hard shot evaluation")

# ── PLOTS v30 ────────────────────────────────────────────────────────
plt_li = plot(training_tsteps[1:n_solved], training_data.y.li[1:n_solved],
              label="Li data", linewidth=2, color=:blue)
plot!(plt_li, training_tsteps[1:n_solved], romero_pred[1, 1:n_solved],
      label="Li Romero", linewidth=2, linestyle=:dash, color=:green)
plot!(plt_li, training_tsteps[1:n_solved], pred_full[1, :],
      label="Li NN", linewidth=2, linestyle=:dot, color=:red)
title!(plt_li, "Li: data vs Romero vs NN")
savefig(plt_li, "li_comparison_v50.png")

plt_ip = plot(training_tsteps[1:n_solved], training_data.y.ip[1:n_solved],
              label="Ip data", linewidth=2, color=:blue)
plot!(plt_ip, training_tsteps[1:n_solved], romero_pred[2, 1:n_solved],
      label="Ip Romero", linewidth=2, linestyle=:dash, color=:green)
plot!(plt_ip, training_tsteps[1:n_solved], pred_full[2, :],
      label="Ip NN", linewidth=2, linestyle=:dot, color=:red)
title!(plt_ip, "Ip: data vs Romero vs NN")
savefig(plt_ip, "ip_comparison_v50.png")

plt_V = plot(training_tsteps[1:n_solved], training_data.y.V[1:n_solved],
             label="V data", linewidth=2, color=:blue)
plot!(plt_V, training_tsteps[1:n_solved], romero_pred[3, 1:n_solved],
      label="V Romero", linewidth=2, linestyle=:dash, color=:green)
plot!(plt_V, training_tsteps[1:n_solved], pred_full[3, :],
      label="V NN", linewidth=2, linestyle=:dot, color=:red)
title!(plt_V, "V: data vs Romero vs NN")
savefig(plt_V, "V_comparison_v50.png")

println("Saved comparison plots.")

# Solve over the full time span in one shot (no multiple shooting)
#=
pred_full = Array(solve(prob_pred_romeroNN, Tsit5(), 
                        p = θ_trained,
                        saveat = training_tsteps,
                        sensealg = ZYG,
                        reltol = 1e-6,
                        abstol = 1e-8,
                        verbose = false))
=#


#Optimization of neural network parameters... do we have any the case of UDE1?

### If the loss is stagnant or starts to increase, then the optimizer is stuck in a minima  
#To avoid this... we have to rerun the model with a different initialization
#Is there a tutorial on achieve robustness?
#What do we mean by robust ... what percentage of all possible parametrizations would converge?
#==
adtype = Optimization.AutoZygote()
optf = Optimization.OptimizationFunction((x,p) -> loss_adjoint_ude1(x), adtype)
optprob_Adam = Optimization.OptimizationProblem(optf, α)
result_Adam = Optimization.solve(optprob_Adam, OptimizationOptimisers.AdamW(0.005), callback = callback, maxiters = 200)

println("Training loss after $(length(losses)) iterations with ADAM: $(losses[end])")

#Train more with BFGS
optprob_BFGS = Optimization.OptimizationProblem(optf, result_Adam.minimzer)
result_BGFS = Optimization.solve(optprob2, Optim.BFGS(initial_stepnorm = 0.01), callback = callback, maxiters = 250)

println("Final training loss for BGFS after $(length(losses)) iterations: $(losses[end])")

#Plots the losses
pl_losses = plot(
                  1:200,
                  losses[1:200],
                  yaxis = :log10,
                  xaxis = :log10,
                  xlabel = "Iterations",
                  ylabel = "Loss",
                  label = "AdamW(0.005, 0.995)",
                  color = :blue
)

plot!(
                251:length(losses),
                losses[251:end],
                yaxis = :log10,
                xaxis = :log10, 
                xlabel = "Iterations",
                ylabel = "Loss",
                label = "BFGS(0.1)",
                color = :red 
)
p_trained = result_BGDS.minimizer

#Match optimized neural network prediction with the underlying data....
#Plot the data and the approximation#
#Û = predict(p_trained, )
=#


