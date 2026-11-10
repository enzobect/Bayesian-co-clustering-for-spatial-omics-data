using JLD2
const FIGURES_DIR = "../../figures"
const COORDS_FILE    = "../../mouse_brain_data/mouse_brain_4700_bin_centers.csv"

function choose_history_and_load()
    history_list = readdir(RESULTS_DIR)
    println("which one?")
    for (i, f) in enumerate(history_list)
        println("$i. $f")
    end
    indx = parse(Int, readline())
    history = load(joinpath(RESULTS_DIR, history_list[indx]))["history"]
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
