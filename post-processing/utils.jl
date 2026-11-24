using JLD2
const FIGURES_DIR     = "../../figures_visium100"
const DIAGNOSTICS_DIR = "../../posterior_loss_visium100"
const COORDS_FILE     = "../Data/Visium_HD_human_tissue_100/human_124504_bin_centers.csv"
const RESULTS_DIR     = "../../potts_history_visium100"   # = paths.path_to_history_files in config.json

function history_files()
    isdir(RESULTS_DIR) || error("History directory does not exist: $(abspath(RESULTS_DIR))")
    files = filter(
        path -> isfile(path) && lowercase(splitext(path)[2]) == ".jld2",
        readdir(RESULTS_DIR; join=true)
    )
    sort!(files)
    isempty(files) && error("No .jld2 history files found in $(abspath(RESULTS_DIR))")
    return files
end

function choose_history_files()
    files = history_files()
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

# Menu for colormap.jl. Returns groups of files; each group is processed together
# (one SALSO run on the pooled draws).
#   0     -> every file on its own
#   1..N  -> that file on its own
#   N+1   -> all files, chains with the same β pooled
function choose_history_groups()
    files = history_files()
    N = length(files)
    println("Which history file should be processed?")
    println("0. All files, each chain separately")
    for (index, path) in enumerate(files)
        println("$index. $(basename(path))")
    end
    println("$(N + 1). All files, chains with the same β merged")

    choice = tryparse(Int, strip(readline()))
    if choice == 0
        return [[f] for f in files]
    elseif choice isa Int && 1 <= choice <= N
        return [[files[choice]]]
    elseif choice == N + 1
        return group_by_beta(files)
    end
    throw(ArgumentError("Please enter a number between 0 and $(N + 1)."))
end

function load_history(path::AbstractString)
    history = load(path)["history"]
    return history
end

# β / seed / replicate stored by the simulation scripts (missing for older files)
function load_run_info(path::AbstractString)
    jldopen(path, "r") do f
        get_key(k) = haskey(f, k) ? f[k] : missing
        return (beta = get_key("beta"), seed = get_key("seed"), replicate = get_key("replicate"))
    end
end

# Group history files by β, so that the replicate chains of the same β can be pooled.
# Files without a stored β (older runs) each form their own group.
function group_by_beta(paths)
    groups = Dict{Any, Vector{String}}()
    for p in paths
        β = load_run_info(p).beta
        push!(get!(groups, ismissing(β) ? p : β, String[]), p)
    end
    # β in increasing order, files without β at the end
    return [sort(v) for (_, v) in sort(collect(groups); by = g -> g[1] isa Real ? (0, g[1], "") : (1, 0.0, g[1]))]
end

function states_to_cluster_mat(history)::Matrix{Int}
    tmp_state = history[1].w
    N = size(tmp_state)[1]
    iters = size(history)[1]
    res = zeros(Int,N, iters)

    for (i,s) in enumerate(history)
        res[:,i] = s.w
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
