# Shared by simulation.jl and simulation_gpu.jl: builds the list of chains to run.
#
# config.json:
#   "hyper_parameters": { "beta": 2.0 }            -> a single β
#   "hyper_parameters": { "beta": [1, 2, 4, 8] }   -> a sweep over several β
#   "sampler": { "n_chains_per_beta": 3 }          -> chains (different seeds) for each β
# (the old key "n_chains" still works as a fallback for "n_chains_per_beta")
using Random, Printf, Dates

struct Chain_Job
    id::Int          # global chain id (1..n_betas*n_chains_per_beta)
    β::Float64
    replicate::Int   # chain index within this β
    seed::UInt32
end

function betas_from_config(config)::Vector{Float64}
    b = config.hyper_parameters.beta
    return b isa AbstractVector ? Float64.(b) : [Float64(b)]
end

function n_chains_per_beta(config)::Int
    s = config.sampler
    return Int(get(s, "n_chains_per_beta", get(s, "n_chains", 1)))
end

"""
One job per (β, replicate), each with its own seed. Seeds are drawn from `master_seed`,
so the whole sweep is reproducible; every chain gets a different seed (also across β).
"""
function chain_jobs(config, master_seed)::Vector{Chain_Job}
    rng = Random.Xoshiro(master_seed)
    betas = betas_from_config(config)
    n = n_chains_per_beta(config)
    seeds = Set{UInt32}()
    jobs = Chain_Job[]
    for β in betas, r in 1:n
        s = rand(rng, UInt32)
        while s in seeds
            s = rand(rng, UInt32)
        end
        push!(seeds, s)
        push!(jobs, Chain_Job(length(jobs) + 1, β, r, s))
    end
    return jobs
end

beta_tag(β) = replace(@sprintf("%g", β), "." => "p")   # 2.5 -> "2p5"

function history_file_path(config, job::Chain_Job)
    dir_path = config.paths.path_to_history_files
    mkpath(dir_path)
    timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
    return joinpath(dir_path, "states_beta$(beta_tag(job.β))_chain$(job.replicate)_$(timestamp).jld2")
end

function print_jobs(jobs::Vector{Chain_Job})
    println("$(length(jobs)) chains:")
    for j in jobs
        println("   chain $(j.id): β = $(j.β), replicate $(j.replicate), seed $(j.seed)")
    end
end
