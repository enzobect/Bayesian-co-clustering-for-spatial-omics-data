using JLD2
const FIGURES_DIR = "../../figures"
const DIAGNOSTICS_DIR = "../../posterior_loss"
const COORDS_FILE    = "../../mouse_brain_data/mouse_brain_4700_bin_centers.csv"
const RESULTS_DIR   = "../../potts_history"

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

function load_history(path::AbstractString)
    history = load(path)["history"]
    return history
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
