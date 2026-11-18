# Simulations on the toy examples.
#
# Runs the vectorised Gibbs sampler of simulation_gpu.jl on the toy examples in ../toy_examples
# for every combination of the parameter values listed in "grid" of the config file
# (by default R = 3, K = 6 and a β sweep), and scores each run against the true spot / gene
# clusterings (adjusted Rand index).
#
# Usage (from any folder):
#     julia -t 4 "Pott's Model simulation/toy_simulation.jl" [config.json]   (default toy_config.json)
# then make the figures with
#     julia post-processing/toy_examples_figures.jl
# -t = Julia threads = number of chains run at the same time.
#
# A GPU is NOT required: without a working CUDA GPU the same vectorised sampler runs on
# CPU arrays (the CUDA package must still be installed: julia -e 'using Pkg; Pkg.add("CUDA")').
# Relative paths in the config are relative to this folder.
#
# Config (toy_config.json):
#   "grid": every key is a list of values (a single number also works); all combinations
#           of R, K, beta (Potts β), mu_0, lambda, alpha_0, beta_0 are run.
#   "examples": null = every "toy_example_*" folder, or a list of folder names.
#   "data.log1p": false. The toy data are already continuous (and can be negative),
#           so log1p must NOT be applied, unlike for the real count data.
#   "sampler": n_iter, burn_in, eval_every (ARI is computed every eval_every iterations
#           after burn-in), n_chains_per_setting (replicates with different seeds),
#           n_concurrent_chains (null = number of Julia threads), save_history.
#
# Output (in paths.output_dir):
#   runs.csv               one row per chain, written as soon as the chain finishes
#   summary_by_setting.csv per (example, parameters): mean over replicates
#   ranking_overall.csv    per parameters: mean / worst over all examples (sorted by mean score)
#   histories/...          thinned w, z traces and ARI traces (.jld2), only if save_history = true
#                          (needed by post-processing/toy_examples_figures.jl)
# score = (ARI of spots + ARI of genes) / 2, averaged over the post burn-in evaluations.
#
# Resuming: runs already in runs.csv are skipped, so if the run is interrupted just
# start it again with the same config (seeds are deterministic).

include("simulation_gpu.jl")   # GPU sampler (does not run its main() when included)
using Statistics

const DEFAULT_TOY_CONFIG = "toy_config.json"
const TUNING_SEED = hash("Toy examples parameter tuning")
const GRID_KEYS = (:R, :K, :beta, :mu_0, :lambda, :alpha_0, :beta_0)

const Params = @NamedTuple{R::Int, K::Int, beta::Float64, mu_0::Float64,
                           lambda::Float64, alpha_0::Float64, beta_0::Float64}

struct Toy_Example
    name::String
    d::Device_Data
    w_true::Vector{Int}   # true spot clusters
    z_true::Vector{Int}   # true gene clusters
end

struct Tuning_Job
    id::Int
    example::String
    params::Params
    replicate::Int
    seed::UInt32
end

# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
# CSV files written by R's write.csv: the first column holds the row names
read_r_matrix(path) = Matrix{Float64}(DataFrame(CSV.File(path))[:, 2:end])
read_r_labels(path) = Int.(DataFrame(CSV.File(path))[:, 2])

function load_toy_example(dir::String, adjacency::Matrix{Int}, use_log1p::Bool)::Toy_Example
    name = basename(dir)
    y = read_r_matrix(joinpath(dir, "samples.csv"))
    w_true = read_r_labels(joinpath(dir, "spot_clustering.csv"))
    z_true = read_r_labels(joinpath(dir, "gene_clustering.csv"))
    N, G = size(y)
    if N != size(adjacency, 1) || N != length(w_true) || G != length(z_true)
        error("$name: samples.csv is $N×$G, but there are $(size(adjacency, 1)) spots in " *
              "adjacency.csv, $(length(w_true)) true spot labels and $(length(z_true)) " *
              "true gene labels. Regenerate the toy examples with toy_examples_generation.R.")
    end
    all(isfinite, y) || error("$name: samples.csv contains non-finite values")
    use_log1p && (y = log1p.(y))
    data = Observed_Data(N, G, y, adjacency)
    return Toy_Example(name, to_device(data), w_true, z_true)
end

function example_dirs(cfg)::Vector{String}
    root = cfg["paths"]["toy_examples_dir"]
    wanted = get(cfg, "examples", nothing)
    names = isnothing(wanted) ?
        sort(filter(n -> startswith(n, "toy_example_") && isdir(joinpath(root, n)), readdir(root))) :
        String.(wanted)
    isempty(names) && error("No toy examples found in $(abspath(root))")
    return [joinpath(root, n) for n in names]
end

# ---------------------------------------------------------------------------
# Grid and jobs
# ---------------------------------------------------------------------------
as_list(x) = x isa AbstractVector ? collect(x) : [x]

function parameter_grid(cfg)::Vector{Params}
    g = cfg["grid"]
    values = [as_list(g[String(k)]) for k in GRID_KEYS]
    grid = Params[]
    for combo in Iterators.product(values...)
        p = Params(combo)
        (p.R < 1 || p.K < 1 || p.R > 255 || p.K > 255) && error("R and K must be in 1:255, got $p")
        push!(grid, p)
    end
    return grid
end

function tuning_jobs(examples::Vector{String}, grid::Vector{Params}, n_rep::Int)::Vector{Tuning_Job}
    rng = Random.Xoshiro(TUNING_SEED)
    jobs = Tuning_Job[]
    for ex in examples, p in grid, r in 1:n_rep
        push!(jobs, Tuning_Job(length(jobs) + 1, ex, p, r, rand(rng, UInt32)))
    end
    return jobs
end

fmt(x::Real) = replace(@sprintf("%g", x), "." => "p")
params_tag(p::Params) = "R$(p.R)_K$(p.K)_beta$(fmt(p.beta))_mu0$(fmt(p.mu_0))_lambda$(fmt(p.lambda))" *
                        "_alpha0$(fmt(p.alpha_0))_beta0$(fmt(p.beta_0))"
run_key(job::Tuning_Job) = "$(job.example)|$(params_tag(job.params))|rep$(job.replicate)"

# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
"""Adjusted Rand index between two labelings (1 = identical partitions, ≈0 = random)."""
function adjusted_rand_index(a::AbstractVector{<:Integer}, b::AbstractVector{<:Integer})
    n = length(a)
    n == length(b) || throw(DimensionMismatch("labelings of different length"))
    ia = Dict(v => i for (i, v) in enumerate(unique(a)))
    ib = Dict(v => i for (i, v) in enumerate(unique(b)))
    C = zeros(Int, length(ia), length(ib))
    for i in 1:n
        C[ia[a[i]], ib[b[i]]] += 1
    end
    c2(x) = x * (x - 1) / 2
    s_ij = sum(c2, C)
    s_a = sum(c2, sum(C, dims=2))
    s_b = sum(c2, sum(C, dims=1))
    expected = s_a * s_b / c2(n)
    max_index = (s_a + s_b) / 2
    return max_index == expected ? 1.0 : (s_ij - expected) / (max_index - expected)
end

n_occupied(labels) = length(unique(labels))

# ---------------------------------------------------------------------------
# One chain
# ---------------------------------------------------------------------------
function run_tuning_chain(ex::Toy_Example, job::Tuning_Job, s)
    p = job.params
    hp = Hyper_Parameters(p.R, p.K, p.mu_0, p.lambda, p.alpha_0, p.beta_0, p.beta, fill(1.0 / p.K, p.K))
    n_iter, burn_in, eval_every = s.n_iter, s.burn_in, s.eval_every

    Random.seed!(job.seed)
    USE_GPU && CUDA.seed!(job.seed)
    d = ex.d
    st = init_state(d, hp)
    logπ = AT(FT.(log.(hp.π)))

    ari_w = Float64[]; ari_z = Float64[]
    occ_w = Int[]; occ_z = Int[]
    history = MCMC_State[]
    t0 = time()
    for i in 1:n_iter
        W = onehot(st.w, hp.R)
        Z = onehot(st.z, hp.K)
        update_μ_and_σ2_gpu!(st, hp, d, W, Z)
        update_z_gpu!(st, d, W, logπ)
        update_w_gpu!(st, hp, d, onehot(st.z, hp.K))

        if i > burn_in && (i - burn_in) % eval_every == 0
            w = Array(st.w)
            z = Array(st.z)
            push!(ari_w, adjusted_rand_index(w, ex.w_true))
            push!(ari_z, adjusted_rand_index(z, ex.z_true))
            push!(occ_w, n_occupied(w))
            push!(occ_z, n_occupied(z))
            s.save_history && push!(history, MCMC_State(UInt8.(w), UInt8.(z), [0.0;;], [0.0;;], Auxiliary_Vars()))
        end
    end
    elapsed = time() - t0

    if s.save_history
        dir = joinpath(s.output_dir, "histories", job.example)
        mkpath(dir)
        history_path = joinpath(dir, "$(params_tag(p))_rep$(job.replicate).jld2")
        mkpath(dirname(history_path))
        jldsave(history_path;
                history=history, params=Dict(pairs(p)), seed=job.seed, replicate=job.replicate,
                burn_in=burn_in, thin=eval_every, w_true=ex.w_true, z_true=ex.z_true,
                ari_w=ari_w, ari_z=ari_z)   # ARI traces, for the convergence figure
    end

    mw, mz = mean(ari_w), mean(ari_z)
    return (; run_key=run_key(job), example=job.example, p..., replicate=job.replicate, seed=job.seed,
            score=(mw + mz) / 2,
            ari_w_mean=mw, ari_w_sd=std(ari_w), ari_w_last=last(ari_w),
            ari_z_mean=mz, ari_z_sd=std(ari_z), ari_z_last=last(ari_z),
            occupied_R_mean=mean(occ_w), occupied_K_mean=mean(occ_z),
            ms_per_iter=1000 * elapsed / n_iter)
end

# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------
const PARAM_COLS = collect(GRID_KEYS)

function write_summaries(runs_path::String, output_dir::String)
    isfile(runs_path) || return
    runs = DataFrame(CSV.File(runs_path))
    nrow(runs) == 0 && return

    by_setting = combine(groupby(runs, [:example; PARAM_COLS]),
        :score => mean => :score,
        :score => (x -> length(x) > 1 ? std(x) : 0.0) => :score_sd_between_chains,
        :ari_w_mean => mean => :ari_w,
        :ari_z_mean => mean => :ari_z,
        :occupied_R_mean => mean => :occupied_R,
        :occupied_K_mean => mean => :occupied_K,
        nrow => :n_chains)
    sort!(by_setting, [:example, order(:score, rev=true)])
    CSV.write(joinpath(output_dir, "summary_by_setting.csv"), by_setting)

    overall = combine(groupby(by_setting, PARAM_COLS),
        :score => mean => :mean_score,
        :score => minimum => :worst_example_score,
        :ari_w => mean => :mean_ari_w,
        :ari_z => mean => :mean_ari_z,
        :occupied_R => mean => :mean_occupied_R,
        :occupied_K => mean => :mean_occupied_K,
        nrow => :n_examples)
    sort!(overall, order(:mean_score, rev=true))
    CSV.write(joinpath(output_dir, "ranking_overall.csv"), overall)

    println("\nTop parameter settings (mean over examples), full table in ranking_overall.csv:")
    show(stdout, first(overall, 10); allcols=true, summary=false, eltypes=false)
    println()
end

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
function main()
    # resolve the config path from where the user launched, then work from this folder
    # so that the relative paths inside the config are always valid
    config_path = isempty(ARGS) ? joinpath(@__DIR__, DEFAULT_TOY_CONFIG) : abspath(ARGS[1])
    isfile(config_path) || error("Config file not found: $config_path")
    cd(@__DIR__)
    cfg = JSON.parsefile(config_path)
    sc = cfg["sampler"]
    output_dir = abspath(cfg["paths"]["output_dir"])
    mkpath(output_dir)
    cp(config_path, joinpath(output_dir, "config_used.json"); force=true)

    s = (n_iter=Int(sc["n_iter"]), burn_in=Int(sc["burn_in"]), eval_every=Int(sc["eval_every"]),
         save_history=Bool(get(sc, "save_history", false)), output_dir=output_dir)
    s.burn_in + s.eval_every <= s.n_iter || error("need burn_in + eval_every <= n_iter")
    n_rep = Int(get(sc, "n_chains_per_setting", 1))
    use_log1p = Bool(get(get(cfg, "data", Dict()), "log1p", false))

    println(USE_GPU ? "Running on GPU: $(CUDA.name(CUDA.device()))" :
                      "No working CUDA GPU found: running vectorised version on CPU")

    adjacency = Matrix{Int}(DataFrame(CSV.File(cfg["paths"]["adjacency"], header=false)))
    examples = Dict{String,Toy_Example}()
    for dir in example_dirs(cfg)
        ex = load_toy_example(dir, adjacency, use_log1p)
        examples[ex.name] = ex
        println("loaded $(ex.name): N = $(ex.d.N), G = $(ex.d.G), true R = $(n_occupied(ex.w_true)), ",
                "true K = $(n_occupied(ex.z_true))")
    end

    grid = parameter_grid(cfg)
    jobs = tuning_jobs(sort(collect(keys(examples))), grid, n_rep)

    runs_path = joinpath(output_dir, "runs.csv")
    done = Set{String}()
    if isfile(runs_path)
        union!(done, String.(DataFrame(CSV.File(runs_path)).run_key))
    end
    todo = filter(j -> !(run_key(j) in done), jobs)
    println("$(length(grid)) parameter settings × $(length(examples)) examples × $n_rep chains = ",
            "$(length(jobs)) chains ($(length(jobs) - length(todo)) already done, $(length(todo)) to run), ",
            "$(s.n_iter) iterations each")

    if !isempty(todo)
        n_workers = clamp(Int(something(get(sc, "n_concurrent_chains", nothing), Threads.nthreads())),
                          1, length(todo))
        println("Running $n_workers chains concurrently ($(Threads.nthreads()) Julia threads)")
        USE_GPU && CUDA.synchronize()

        write_lock = ReentrantLock()
        n_finished = Threads.Atomic{Int}(0)
        queue = Channel{Tuning_Job}(length(todo))
        foreach(j -> put!(queue, j), todo)
        close(queue)
        t_start = time()

        @sync for _ in 1:n_workers
            Threads.@spawn for job in queue
                row = try
                    run_tuning_chain(examples[job.example], job, s)
                catch err
                    @warn "chain $(job.id) failed" job.example job.params exception = (err, catch_backtrace())
                    nothing
                end
                k = Threads.atomic_add!(n_finished, 1) + 1
                lock(write_lock) do
                    if row === nothing
                        open(joinpath(output_dir, "failed_runs.txt"), "a") do io
                            println(io, run_key(job))
                        end
                    else
                        CSV.write(runs_path, [row]; append=isfile(runs_path))
                    end
                    el = time() - t_start
                    msg = row === nothing ? "FAILED" :
                          "score $(round(row.score, digits=3)) (ARI spots $(round(row.ari_w_mean, digits=3)), " *
                          "genes $(round(row.ari_z_mean, digits=3)), $(round(row.ms_per_iter, digits=1)) ms/iter)"
                    println("[$k/$(length(todo))] $(job.example) $(params_tag(job.params)) rep $(job.replicate): $msg",
                            "  ETA $(round((length(todo) - k) * el / k / 60, digits=1)) min")
                end
            end
        end
    end

    write_summaries(runs_path, output_dir)
    println("Results in $(abspath(output_dir))")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
