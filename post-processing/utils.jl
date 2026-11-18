using JLD2
const FIGURES_DIR     = "../../figures_visium100"
const DIAGNOSTICS_DIR = "../../posterior_loss_visium100"
const COORDS_FILE     = "../Data/Visium_HD_human_tissue_100/human_124504_bin_centers.csv"

function history_files(dir = RESULTS_DIR)
    isdir(dir) || error("History directory does not exist: $(abspath(dir))")
    files = filter(
        path -> isfile(path) && lowercase(splitext(path)[2]) == ".jld2",
        readdir(dir; join=true)
    )
    sort!(files)
    isempty(files) && error("No .jld2 history files found in $(abspath(dir))")
    return files
end

function choose_history_files(dir = RESULTS_DIR)
    files = history_files(dir)
    println("Which history file should be processed?")
    println("0. All files")
    for (index, path) in enumerate(files)
        println("$index. $(basename(path))")
    end

    choice = try
        parse(Int, strip(readline()))
    catch error
        throw(ArgumentError("Please enter a number between 0 and $(length(files))."))
    end

    if choice == 0
        return files
    elseif 1 <= choice <= length(files)
        return [files[choice]]
    end

    throw(ArgumentError("Please enter a number between 0 and $(length(files))."))
end

function load_history(path::AbstractString)
    history = load(path)["history"]
    return history
end

"""
    load_chain(path, burn_in_frac, thin)

Everything post-processing needs from one chain file, as a NamedTuple:

    path, info        file and (beta, seed, replicate)
    n_iter, burn      iterations in the chain / discarded as burn-in
    kept              iterations kept after burn-in and thinning
    W :: N × S UInt8  spot labels at `kept`
    Z :: G × S UInt8  gene labels at `kept`
    mu, sigma2        R × K × n_iter Float32, whole chain (the caterpillar wide view shows the burn-in too)
    occupied          R × K BitMatrix: block (r, k) non-empty at every post-burn-in iteration
                      (empty blocks are draws from the prior, not from the data)

Reads the format of simulation_gpu.jl (`w_samples`, ...: seconds to load). Old files with
`history` still work (slow: ~13 min for 124504 spots) but have no μ / σ²: mu = sigma2 = occupied = nothing.
"""
function load_chain(path::AbstractString, burn_in_frac::Real, thin::Int)
    info = load_run_info(path)
    window(n) = (b = floor(Int, burn_in_frac * n); (b, (b + 1):thin:n))

    if !jldopen(f -> haskey(f, "w_samples"), path, "r")
        println("   old `history` format: slow to load and no μ / σ² stored (re-run simulation_gpu.jl)")
        history = load_history(path)
        n = length(history)
        burn, kept = window(n)
        return (path = path, info = info, n_iter = n, burn = burn, kept = kept,
                W = UInt8.(states_to_cluster_mat(history, kept; field = :w)),
                Z = UInt8.(states_to_cluster_mat(history, kept; field = :z)),
                mu = nothing, sigma2 = nothing, occupied = nothing)
    end

    jldopen(path, "r") do f
        w = f["w_samples"]
        z = f["z_samples"]
        n = size(w, 2)
        burn, kept = window(n)
        post = (burn + 1):n
        has_params = haskey(f, "mu_samples")
        mu = has_params ? f["mu_samples"] : nothing
        sigma2 = has_params ? f["sigma2_samples"] : nothing
        occupied = has_params ? block_occupancy(w, z, post, size(mu, 1), size(mu, 2)) : nothing
        return (path = path, info = info, n_iter = n, burn = burn, kept = kept,
                W = w[:, kept], Z = z[:, kept], mu = mu, sigma2 = sigma2, occupied = occupied)
    end
end

# variation of information between two partitions (natural log; 0 = identical, label numbering irrelevant)
function vi_distance(a::AbstractVector{<:Integer}, b::AbstractVector{<:Integer})
    n = length(a)
    @assert length(b) == n
    joint = Dict{Tuple{Int,Int},Int}()
    ca = Dict{Int,Int}(); cb = Dict{Int,Int}()
    for i in 1:n
        joint[(a[i], b[i])] = get(joint, (a[i], b[i]), 0) + 1
        ca[a[i]] = get(ca, a[i], 0) + 1
        cb[b[i]] = get(cb, b[i], 0) + 1
    end
    h(counts) = -sum(c / n * log(c / n) for c in counts)
    return 2 * h(values(joint)) - h(values(ca)) - h(values(cb))
end

# block (r, k) occupied =spot cluster r and gene cluster k both non-empty at every iteration in `iters`
function block_occupancy(w::AbstractMatrix, z::AbstractMatrix, iters, R::Int, K::Int)::BitMatrix
    always_nonempty(labels, L) = begin
        ok = trues(L)
        counts = zeros(Int, L)
        for t in iters
            fill!(counts, 0)
            @inbounds for i in axes(labels, 1)
                counts[labels[i, t]] += 1
            end
            ok .&= counts .> 0
        end
        ok
    end
    return always_nonempty(w, R) .& always_nonempty(z, K)'
end

"""
Chains grouped by β (sorted), for pooling. Files with the same seed are the same chain
(e.g. re-runs of the same config): only the newest one is kept, so no chain is counted twice.
"""
function group_chains_by_beta(chains)
    by_seed = Dict{Any,Any}()
    for c in chains
        key = (c.info.beta, c.info.seed)
        if haskey(by_seed, key) && !ismissing(c.info.seed)
            old = by_seed[key]
            newer = basename(c.path) > basename(old.path) ? c : old   # file names end in a timestamp
            println("   WARNING: $(basename(old.path)) and $(basename(c.path)) have the same seed: ",
                    "pooling only $(basename(newer.path))")
            by_seed[key] = newer
        else
            by_seed[ismissing(c.info.seed) ? (c.info.beta, c.path) : key] = c
        end
    end
    groups = Dict{Any,Vector{Any}}()
    for c in values(by_seed)
        push!(get!(groups, c.info.beta, Any[]), c)
    end
    return [(β, sort(groups[β]; by = c -> basename(c.path))) for β in sort(collect(keys(groups)))]
end

# β / seed / replicate stored by the simulation scripts (missing for older files)
function load_run_info(path::AbstractString)
    jldopen(path, "r") do f
        get_key(k) = haskey(f, k) ? f[k] : missing
        return (beta = get_key("beta"), seed = get_key("seed"), replicate = get_key("replicate"))
    end
end

# items × selected iterations. `field` = :w (spots) or :z (genes).
# Only the iterations in `cols` are copied: with 124504 spots the full chain would not fit in RAM.
function states_to_cluster_mat(history, cols = eachindex(history); field::Symbol = :w)::Matrix{Int}
    N = length(getfield(history[1], field))
    res = zeros(Int, N, length(cols))
    for (i, c) in enumerate(cols)
        res[:, i] = getfield(history[c], field)
    end
    return res
end

function coclustering_distance(group_iter::Matrix{Int}, cols)
    N = size(group_iter, 1)
    ns = length(cols)
    T = ns <= typemax(UInt16) ? UInt16 : UInt32
    C = zeros(T, N, N)
    for c in cols
        lab = @view group_iter[:, c]
        members = [Int[] for _ in 1:(maximum(lab) + 1)]
        for i in 1:N
            push!(members[lab[i] + 1], i)
        end
        for idx in members                      # idx is sorted -> column-wise access
            @inbounds for j in idx, i in idx
                C[i, j] += one(T)
            end
        end
    end
    D = Matrix{Float64}(undef, N, N)
    @inbounds for j in 1:N, i in 1:N
        D[i, j] = 1.0 - C[i, j] / ns
    end
    return D, ns
end
