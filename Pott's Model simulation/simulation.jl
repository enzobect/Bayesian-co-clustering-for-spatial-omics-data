include("Potts_MCMC_sampler.jl")
using CSV
using JSON
using JLD2
using Dates
using DataFrames
using Term.Progress
# %%
const CONFIG_FILE_PATH = "config.json"
const SEED = hash("MCMC sampler for the Pott's Model")
# %%


@inline function load_data(config)::Observed_Data
    y_data_path = config.paths.path_to_values
    adj_path = config.paths.path_to_adjecency

    obs_values= DataFrame(CSV.File(y_data_path))
    adjecency = DataFrame(CSV.File(adj_path, header=false))

    data = Observed_Data(DataFrames.nrow(obs_values),
                         DataFrames.ncol(obs_values),
                            log1p.(Matrix(obs_values)),
                            Matrix(adjecency))
    return data
end
# %%
@inline function load_hyper_parameters(config)::Hyper_Parameters
    π = config.hyper_parameters.pi
    if isnothing(π) # if weights not specified default to unif. dis'n
        π = ones(config.hyper_parameters.K)
        π /= sum(π)
    end

    tmp = config.hyper_parameters
    return Hyper_Parameters(tmp.R, tmp.K,
                            tmp.mu_0, tmp.lambda, tmp.alpha_0, tmp.beta_0,
                            tmp.beta,
                            π)
end
# %%
@inline function save_history(config, history::Vector{MCMC_State},id::Int=nothing)
    dir_path = config.paths.path_to_history_files
    timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")

    if id !== nothing
        file_name = "states_$(timestamp)_$id.jld2"
    else
        file_name = "states_$timestamp.jld2"
    end
    mkpath(dir_path)
    jldsave(joinpath(dir_path, file_name), history=history)
end
# %%

# %%

@inline function init_state(N::Int, G::Int, hyper_parameters::Hyper_Parameters)::MCMC_State
    R = hyper_parameters.R
    K = hyper_parameters.K
    return MCMC_State(sample(1:R,N), sample(1:K,G),
                      ones(R,K), ones(R,K),
                      Auxiliary_Vars())
end

function summarize_state(state::MCMC_State)::MCMC_State
    return MCMC_State(deepcopy(state.w),deepcopy(state.z), [0;;], [0;;], Auxiliary_Vars())
end

function run_MCMC_single_chain(config, data::Observed_Data, hp::Hyper_Parameters, seed::Int, chain_id::Int, job::ProgressJob)
    Random.seed!(seed)
    state::MCMC_State = init_state(data.N,data.G,hp)
    history = MCMC_State[] # TODO: takes too much space

    for i in 1:config.sampler.n_iter
        Progress.update!(job)
        update_μ_and_σ2!(state,hp, data)
        update_z!(state,hp,data)
        update_w!(state,hp,data)

        push!(history, summarize_state(state))
        # TODO: save the history frequently to support early termination
    end


    save_history(config,history, chain_id)
end
function run_MCMC_multiple_chains(config, data::Observed_Data, hp::Hyper_Parameters, seeds::Vector{Int}, pbar::Progress.ProgressBar, pb_lock::ReentrantLock)
    for (i,seed) in enumerate(seeds)
        job = nothing
        @lock pb_lock begin
            job = addjob!(pbar, N = convert(Int,config.sampler.n_iter), description="chain $i at worker #$(Threads.threadid() - 1)")
        end
        run_MCMC_single_chain(config, data,hp, seed, Threads.threadid(), job)
        @lock pb_lock begin
            removejob!(pbar, job)
        end
    end

end
# %%
function main()
    config = JSON.parsefile(CONFIG_FILE_PATH)
    Random.seed!(SEED)

    hyper_parameters::Hyper_Parameters = load_hyper_parameters(config)

    data::Observed_Data = load_data(config)

    n_threads = Threads.nthreads()

    n_chains = config.sampler.n_chains

    ids_to_seeds = Dict{Int,Vector{Int}}()
    for i in 1:n_threads
        ids_to_seeds[i] = Int[]
    end
    for i in 1:n_chains
        push!(ids_to_seeds[i % n_threads + 1], Random.rand(Int,1)[1]) # TODO: fix the seed
    end

    pbar = ProgressBar()
    pb_update_lock = ReentrantLock()
    with(pbar) do # the block makes sure the bars update while the threads are running
        @sync begin # ensure the process doesn't terminate before all chains finish execution
            for i in 1:n_threads
                Threads.@spawn run_MCMC_multiple_chains(config, data, hyper_parameters, ids_to_seeds[i], pbar, pb_update_lock)
            end
        end
    end
end

# %%
main()
