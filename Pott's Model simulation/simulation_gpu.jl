# GPU version of the Potts co-clustering Gibbs sampler.
#
# Usage (from this folder):
#     julia -t 4 simulation_gpu.jl
# (-t = Julia threads; chains run concurrently on the GPU, see "n_concurrent_chains")
#
# One-time setup:
#     julia -e 'using Pkg; Pkg.add("CUDA")'
#
# Same config.json and same output format (.jld2 files with `history`) as
# simulation.jl, so the post-processing scripts work unchanged.
# If no working CUDA GPU is found, it falls back to CPU arrays (still
# vectorised, so usually faster than the original loops).
#
# Differences from simulation.jl:
#   * all likelihood terms are computed with matrix products (one-hot cluster matrices)
#   * z | rest : all genes are conditionally independent given w -> sampled in one shot
#   * w | rest : chromatic Gibbs. Spots are split into colour classes such that no two
#                spots in a class are neighbours; each class is sampled in parallel.
#                (Still a valid Gibbs sampler for the Potts model.)
#   * categorical draws use the Gumbel-max trick
#   * μ, σ² (only R×K values) are sampled on the CPU from sufficient statistics
#     computed on the GPU
#   * several chains run concurrently on the single GPU: each chain is a Julia task, and
#     CUDA.jl gives every task its own stream and its own (task-local) RNG, so kernels of
#     different chains overlap and the CPU parts (μ, σ² sampling) run on different threads.
#     "n_concurrent_chains" in config.json (default: number of Julia threads).
#
# β sweep: set "beta" in config.json to a list (e.g. [1, 2, 4, 8]) and
# "n_chains_per_beta" in "sampler"; every chain gets its own seed. Each output file
# stores `history`, `beta`, `seed` and `replicate`, and β is also in the file name.

include("Potts_MCMC_sampler.jl")   # Observed_Data, Hyper_Parameters, MCMC_State, Auxiliary_Vars
include("chains_setup.jl")         # Chain_Job, chain_jobs, history_file_path
using CUDA
using CSV, JSON, JLD2, Dates, DataFrames
using Random, Distributions, LinearAlgebra

const CONFIG_FILE_PATH = "config.json"
const SEED = hash("MCMC sampler for the Pott's Model")

const FT = Float32
const USE_GPU = CUDA.functional()
const AT = USE_GPU ? CuArray : Array

# ---------------------------------------------------------------------------
# Loading (same as simulation.jl)
# ---------------------------------------------------------------------------
function load_data(config)::Observed_Data
    obs_values = DataFrame(CSV.File(config.paths.path_to_values))
    adjecency  = DataFrame(CSV.File(config.paths.path_to_adjecency, header=false))
    return Observed_Data(DataFrames.nrow(obs_values), DataFrames.ncol(obs_values),
                         log1p.(Matrix(obs_values)), Matrix(adjecency))
end

function load_hyper_parameters(config, β::Real)::Hyper_Parameters
    π = config.hyper_parameters.pi
    if isnothing(π)
        π = ones(config.hyper_parameters.K)
        π /= sum(π)
    end
    tmp = config.hyper_parameters
    return Hyper_Parameters(tmp.R, tmp.K, tmp.mu_0, tmp.lambda, tmp.alpha_0, tmp.beta_0,
                            Float64(β), π)
end

function save_history(config, history::Vector{MCMC_State}, job::Chain_Job)
    path = history_file_path(config, job)
    jldsave(path; history=history, beta=job.β, seed=job.seed, replicate=job.replicate)
    println("chain $(job.id) (β = $(job.β)): saved $(abspath(path))")
end

# ---------------------------------------------------------------------------
# Device data
# ---------------------------------------------------------------------------
struct Device_Data{M<:AbstractMatrix{FT}, V<:AbstractVector{Int32}, IM<:AbstractMatrix{Int32}}
    N::Int
    G::Int
    Y::M                 # N×G  log1p counts
    Y2::M                # N×G  squared values
    colors::Vector{V}    # spots of each colour class
    colors_adj::Vector{IM}  # adjacency rows of those spots; padding 0 -> N+1 (dummy spot)
end

"""
Greedy graph colouring: no two neighbouring spots get the same colour.
Neighbourhoods are symmetrised, so it is safe even if the adjacency file is not.
A hexagonal grid needs 3 colours; greedy uses at most 7.
"""
function color_graph(adj::Matrix{Int}, N::Int)::Vector{Vector{Int}}
    nbrs = [Int[] for _ in 1:N]
    for n in 1:N, m in view(adj, n, :)
        if m != 0 && m != n
            push!(nbrs[n], m)
            push!(nbrs[m], n)
        end
    end
    color = zeros(Int, N)
    for n in 1:N
        used = Set(color[m] for m in nbrs[n])
        c = 1
        while c in used
            c += 1
        end
        color[n] = c
    end
    return [findall(==(c), color) for c in 1:maximum(color)]
end

function to_device(data::Observed_Data)
    classes = color_graph(data.adjecency, data.N)
    adj_pad = replace(data.adjecency, 0 => data.N + 1)
    Y = AT(FT.(data.y))
    return Device_Data(data.N, data.G, Y, Y .^ 2,
                       [AT(Int32.(idx)) for idx in classes],
                       [AT(Int32.(adj_pad[idx, :])) for idx in classes])
end

mutable struct Device_State{V<:AbstractVector{Int32}, M<:AbstractMatrix{FT}}
    w::V    # N spot labels
    z::V    # G gene labels
    μ::M    # R×K
    σ2::M   # R×K
end

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
# one-hot matrix: labels (length n) -> n×L, label 0 gives an all-zero row
onehot(labels::AbstractVector{Int32}, L::Int) = FT.(labels .== AT(Int32.(1:L))')

# Sample one category per row of a matrix of (unnormalised) log-probabilities
function gumbel_argmax(logp::AbstractMatrix)
    U = rand!(similar(logp))
    U .= clamp.(U, floatmin(FT), prevfloat(one(FT)))
    noisy = logp .- log.(.-log.(U))
    return Int32.(vec(map(I -> I[2], argmax(noisy, dims=2))))
end

# Terms that appear in both z and w full conditionals:
#   Σ (y-μ)²/σ² + n log σ²  =  Σy²·A - 2Σy·B + n·C
function precision_terms(st::Device_State)
    A = 1 ./ st.σ2
    B = st.μ .* A
    C = st.μ .^ 2 .* A .+ log.(st.σ2)
    return A, B, C
end

# ---------------------------------------------------------------------------
# Gibbs updates
# ---------------------------------------------------------------------------
function update_μ_and_σ2_gpu!(st::Device_State, hp::Hyper_Parameters, d::Device_Data, W, Z)
    # block sufficient statistics (R×K), computed on the GPU, moved to CPU in Float64
    S1 = Float64.(Array(W' * (d.Y * Z)))
    S2 = Float64.(Array(W' * (d.Y2 * Z)))
    nr = Float64.(Array(vec(sum(W, dims=1))))
    nk = Float64.(Array(vec(sum(Z, dims=1))))
    m = nr * nk'

    μ = similar(S1)
    σ2 = similar(S1)
    for i in eachindex(S1)
        if m[i] == 0  # empty block: posterior = prior
            λs, μs, αs, βs = hp.λ, hp.μ0, hp.α0, hp.β0
        else
            ȳ = S1[i] / m[i]
            ss = max(S2[i] - m[i] * ȳ^2, 0.0)
            λs = hp.λ + m[i]
            μs = (hp.λ * hp.μ0 + m[i] * ȳ) / λs
            αs = hp.α0 + m[i] / 2
            βs = hp.β0 + ss / 2 + (hp.λ * m[i] / (2 * λs)) * (ȳ - hp.μ0)^2
        end
        σ2[i] = rand(InverseGamma(αs, βs))
        μ[i] = rand(Normal(μs, sqrt(σ2[i] / λs)))
    end
    st.μ = AT(FT.(μ))
    st.σ2 = AT(FT.(σ2))
    return nothing
end

function update_z_gpu!(st::Device_State, d::Device_Data, W, logπ)
    A, B, C = precision_terms(st)
    nr = vec(sum(W, dims=1))     # R
    T1 = W' * d.Y                # R×G
    T2 = W' * d.Y2               # R×G
    logp = logπ' .- FT(0.5) .* (T2' * A .- 2 .* (T1' * B) .+ (nr' * C))   # G×K
    st.z = gumbel_argmax(logp)
    return nothing
end

function update_w_gpu!(st::Device_State, hp::Hyper_Parameters, d::Device_Data, Z)
    A, B, C = precision_terms(st)
    nk = vec(sum(Z, dims=1))     # K
    S1 = d.Y * Z                 # N×K
    S2 = d.Y2 * Z                # N×K
    # data part of the log full conditional, N×R (does not depend on w)
    D = FT(-0.5) .* (S2 * A' .- 2 .* (S1 * B') .+ (C * nk)')
    β = FT(hp.β)
    pad = AT(Int32[0])

    for c in shuffle(1:length(d.colors))
        idx = d.colors[c]
        adj = d.colors_adj[c]
        Wpad = onehot(vcat(st.w, pad), hp.R)   # (N+1)×R, last row = 0 (padding)
        counts = Wpad[adj[:, 1], :]
        for j in 2:size(adj, 2)
            counts = counts .+ Wpad[adj[:, j], :]
        end
        st.w[idx] = gumbel_argmax(D[idx, :] .+ β .* counts)
    end
    return nothing
end

# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
function init_state(d::Device_Data, hp::Hyper_Parameters)
    return Device_State(AT(Int32.(rand(1:hp.R, d.N))), AT(Int32.(rand(1:hp.K, d.G))),
                        AT(ones(FT, hp.R, hp.K)), AT(ones(FT, hp.R, hp.K)))
end

summarize(st::Device_State) =
    MCMC_State(UInt8.(Array(st.w)), UInt8.(Array(st.z)), [0.0;;], [0.0;;], Auxiliary_Vars())

    # Save cluster labels, block means, and block variances.
# Array transfers the values to CPU memory and creates a snapshot.
full_state(st::Device_State) =
    MCMC_State(
        UInt8.(Array(st.w)),
        UInt8.(Array(st.z)),
        Float64.(Array(st.μ)),
        Float64.(Array(st.σ2)),
        Auxiliary_Vars()
    )

function run_chain_gpu(config, d::Device_Data, hp::Hyper_Parameters, job::Chain_Job, n_iter::Int)
    Random.seed!(job.seed)
    USE_GPU && CUDA.seed!(job.seed)
    
    st = init_state(d, hp)
    logπ = AT(FT.(log.(hp.π)))
    history = MCMC_State[]
    
    # Choose which information to preserve at every iteration.
    save_parameters = Bool(get(config.sampler, "save_mu_sigma2", false))
    snapshot = save_parameters ? full_state : summarize

    t0 = time()
    for i in 1:n_iter
        W = onehot(st.w, hp.R)
        Z = onehot(st.z, hp.K)
        update_μ_and_σ2_gpu!(st, hp, d, W, Z)
        update_z_gpu!(st, d, W, logπ)
        update_w_gpu!(st, hp, d, onehot(st.z, hp.K))
        
        # Store a separate snapshot of this iteration.
        push!(history, snapshot(st))
        if i % 100 == 0 || i == n_iter
            el = time() - t0
            println("chain $(job.id) (β = $(hp.β), rep $(job.replicate)): iter $i/$n_iter  ($(round(el / i * 1000, digits=1)) ms/iter, ",
                    "ETA $(round((n_iter - i) * el / i / 60, digits=1)) min)")
        end
    end
    save_history(config, history, job)
end

function main()
    config = JSON.parsefile(CONFIG_FILE_PATH)
    data = load_data(config)
    d = to_device(data)

    println(USE_GPU ? "Running on GPU: $(CUDA.name(CUDA.device()))" :
                      "No working CUDA GPU found: running vectorised version on CPU")
    println("N = $(d.N) spots, G = $(d.G) genes, $(length(d.colors)) colour classes, precision $FT")

    n_iter = Int(config.sampler.n_iter)
    jobs = chain_jobs(config, SEED)
    print_jobs(jobs)

    n_workers = clamp(Int(get(config.sampler, "n_concurrent_chains", Threads.nthreads())), 1, length(jobs))
    println("Running $n_workers chains concurrently ($(Threads.nthreads()) Julia threads)")
    USE_GPU && CUDA.synchronize()   # device data ready before other tasks/streams read it

    queue = Channel{Chain_Job}(length(jobs))
    foreach(j -> put!(queue, j), jobs)
    close(queue)
    @sync for _ in 1:n_workers
        Threads.@spawn for job in queue
            # each chain seeds its own task-local RNGs (CPU and GPU) inside run_chain_gpu
            hp = load_hyper_parameters(config, job.β)
            run_chain_gpu(config, d, hp, job, n_iter)
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
