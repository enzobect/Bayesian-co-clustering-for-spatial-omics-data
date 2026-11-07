include("Potts_MCMC_sampler.jl")
using CSV
using JSON
using JLD2
using Dates
using DataFrames
using ProgressMeter
# %%
const CONFIG_FILE_PATH = "config.json"
const SEED = hash("MCMC sampler for the Pott's Model")
# %%


@inline function load_data(config)::Observed_Data
    y_data_path = config.paths.path_to_values
    adj_path = config.paths.path_to_adjecency

    obs_values= DataFrame(CSV.File(y_data_path))
    adjecency = DataFrame(CSV.File(adj_path))

    data = Observed_Data(DataFrames.nrow(obs_values),
                         DataFrames.ncol(obs_values),
                            Matrix(obs_values),
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
@inline function save_history(config, history::Vector{MCMC_State})
    dir_path = config.paths.path_to_history_files
    timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
    file_name = "states_$timestamp.jld2"

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
# %%
function main()
    config = JSON.parsefile(CONFIG_FILE_PATH)

    hyper_parameters::Hyper_Parameters = load_hyper_parameters(config)

    data::Observed_Data = load_data(config)

    # TODO: initialize The MCMC state
    state::MCMC_State = init_state(data.N,data.G,hyper_parameters)
    history = MCMC_State[] # TODO: takes too much space
    Random.seed!(SEED)

    @showprogress for i in 1:config.sampler.n_iter
        # sleep(.001)
        update_μ_and_σ2!(state,hyper_parameters)
        update_z!(state,hyper_parameters)
        update_w!(state,hyper_parameters,data)

        push!(history, deepcopy(state))
        # TODO: save the history frequently to support early termination
    end


    save_history(config,history)
end

# %%
main()
