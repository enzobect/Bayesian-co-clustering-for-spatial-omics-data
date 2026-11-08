using LinearAlgebra
using Statistics, Distributions, Random
struct Observed_Data
    #dimensions
    N::Int # number of points
    G::Int # number of genes
    # data
    y::Matrix{Float64} # N × G matrix y[i,j] = counts at spot i for gene j
    adjecency::Matrix{Int} # N × 6 matrix. If a node has less than 6 neighbors,
                           # a suffix of the elements in its row are 0
                           # vertices should be one-indexed
end

mutable struct Auxiliary_Vars # If, for computational efficiency, makes sense to store extra values.

end

mutable struct MCMC_State
    w::Vector{Int}
    z::Vector{Int}
    μ::Matrix{Float64}
    σ2::Matrix{Float64}
    auxiliary::Auxiliary_Vars
end

struct Hyper_Parameters
    # number of clusters
    R::Int # spots
    K::Int # genes
    # μ and σ2
    # Normal Inv. Gamma dist'n parameters
    μ0::Float64
    λ::Float64
    α0::Float64
    β0::Float64
    # w
    # weight for the Pott's model
    β::Float64
    # z
    # probabilities for the multinomial prior
    π::Vector{Float64}
end

function update_μ_and_σ2!(
    state::MCMC_State,
    hyper_parameters::Hyper_Parameters,
    data::Observed_Data
)
    R = hyper_parameters.R
    K = hyper_parameters.K

    μ0 = hyper_parameters.μ0
    λ = hyper_parameters.λ
    α0 = hyper_parameters.α0
    β0 = hyper_parameters.β0

    cluster_to_spots = get_spots_of_clusters(state, data.N, R)
    cluster_to_genes = get_genes_of_clusters(state, data.G, K)

    for r in 1:R
        for k in 1:K
            spots = cluster_to_spots[r]
            genes = cluster_to_genes[k]

            # Number of observations in this block
            m = length(spots) * length(genes)

            if m == 0
                # Empty block: posterior equals the prior
                λ_star = λ
                μ_star = μ0
                α_star = α0
                β_star = β0
            else
                values = @view data.y[spots, genes]

                block_mean = mean(values)
                squared_deviations =
                    sum(x -> (x - block_mean)^2, values)

                # Posterior parameters
                λ_star = λ + m
                μ_star = (λ * μ0 + m * block_mean) / λ_star
                α_star = α0 + m / 2

                β_star = β0 +
                         squared_deviations / 2 +
                         (λ * m / (2 * λ_star)) *
                         (block_mean - μ0)^2
            end

            # Sample variance first, then mean conditional on variance
            state.σ2[r, k] = rand(InverseGamma(α_star, β_star))

            state.μ[r, k] = rand(
                Normal(
                    μ_star,
                    sqrt(state.σ2[r, k] / λ_star)
                )
            )
        end
    end

    return nothing
end

@inline function get_spots_of_clusters(state::MCMC_State,N::Int, R::Int)::Dict{Int, Vector{Int}}
    # construct the data structure
    res = Dict{Int, Vector{Int}}()
    for r in 1:R
        res[r] = Int[]
    end

    #fill in the inverse map
    for n in 1:N
        push!(res[state.w[n]], n)
    end

    return res
end
"""
returns a dictionary from spot cluster idx r to a 2-element vector
the vector consists of

    1. ∑y_n,i for i in cluster r
    2. ∑(y_n,i)^2 for i in cluster r
"""
@inline function get_moments_of_values_by_spot_cluster(data::Observed_Data, cluster_to_spots::Dict{Int, Vector{Int}}, g::Int,R::Int)::Dict{Int,Vector{Float64}}
    moment = Dict{Int,Vector{Float64}}()
    for r in 1:R
        moment[r] = Float64[0,0]
        values = @view data.y[cluster_to_spots[r], g]
        moment[r][1] = sum(values)
        moment[r][2] = sum(values .^ 2)
    end
    return moment
end

function update_z!(state::MCMC_State, hyper_parameters::Hyper_Parameters, data::Observed_Data)

  N = data.N
  G = data.G
  K = hyper_parameters.K
  R = hyper_parameters.R
  π = hyper_parameters.π

  cluster_to_spot = get_spots_of_clusters(state, N, R)

  # For each gene g in {1,...,G}
  for g in 1:G
    log_probabilities = zeros(K) 

    moment_by_cluster = get_moments_of_values_by_spot_cluster(data, cluster_to_spot, g, R)

    for k in 1:K
      # Initialize log probabilities with the prior probabilities
      log_probabilities[k] = log(π[k])

      # Evaluation of the posterior distribution (Equation 11), with optimization also used in update_w!()
      for r in 1:R
        n_values = length(cluster_to_spot[r])
        cluster_mean = state.μ[r,k]
        cluster_variance = state.σ2[r,k]

        log_probabilities[k] += -0.5 *
                            (((moment_by_cluster[r][2] - 2 * cluster_mean * moment_by_cluster[r][1] + cluster_mean^2 * n_values)
                              / cluster_variance)
                              + log(cluster_variance) * n_values
                            ) 
      end
    end
  
    max_log_probabilities = maximum(log_probabilities)

    # Normalize the probabilities and take the exponential to get the actual probabilities
    # .- does the vector-scalar subtraction and exp. does the element-wise exponential (instead of the matrix exponential)
    normalized_probabilities = exp.(log_probabilities .- max_log_probabilities) 
    normalized_probabilities /= sum(normalized_probabilities)

    # Sample a new cluster assignment from the Multinomial with the computed probabilities
    state.z[g] = rand(Categorical(normalized_probabilities))

  end

end

# TODO: instead of computing this every time for w|res, compute and upate in z|res and store in the aux. object
@inline function get_genes_of_clusters(state::MCMC_State,G::Int, K::Int)::Dict{Int, Vector{Int}}
    # construct the data structure
    res = Dict{Int, Vector{Int}}()
    for c in 1:K
        res[c] = Int[]
    end

    #fill in the inverse map
    for g in 1:G
        push!(res[state.z[g]], g)
    end

    return res
end
"""
returns a dictionary from gene cluster idx c to a 2-element vector
the vector consists of

    1. ∑y_n,i for i in cluster c
    2. ∑(y_n,i)^2 for in cluster c
"""
@inline function get_moments_of_values_by_gene_cluster(data::Observed_Data, cluster_to_genes::Dict{Int, Vector{Int}}, n::Int,K::Int)::Dict{Int,Vector{Float64}}
    moment = Dict{Int,Vector{Float64}}()
    for c in 1:K
        moment[c] = Float64[0,0]
        values = @view data.y[n, cluster_to_genes[c]]
        moment[c][1] = sum(values)
        moment[c][2] = sum(values .^ 2)
    end
    return moment
end
function update_w!(state::MCMC_State, hyper_parameters::Hyper_Parameters, data::Observed_Data)

  R = hyper_parameters.R
  N = data.N
  G = data.G
  β = hyper_parameters.β
  K = hyper_parameters.K
  # Generate a permutation σ of J = {1,2,...,N}
  permutation = randperm(N)

  cluster_to_gene = get_genes_of_clusters(state, G, K)

  # For j in 1,...,N
  for n in permutation
    log_probabilities = zeros(R)

    moment_by_cluster = get_moments_of_values_by_gene_cluster(data, cluster_to_gene, n, K)

    for r in 1:R
      # Count the number of neighbors of spot n that are assigned to cluster r (δ(w_j, w_j') in Equation 6)
      number_of_neighbors_r = 0
      for neighbor in data.adjecency[n,:]
        if neighbor != 0 && state.w[neighbor] == r
          number_of_neighbors_r += 1
        end
      end

      # Evaluation of the posterior distribution h_j (w_j) (Equation 12)
      data_likelihood = 0.0
      for c in 1:K
          n_values = length(cluster_to_gene[c])
          cluster_mean = state.μ[r,c]
          cluster_variance = state.σ2[r,c]
          
          data_likelihood += -0.5 * # ∑(vi - μ)^2 = ∑vi^2 - 2μ∑vi + μ^2×#{g in cluster c}
                            (((moment_by_cluster[c][2] - 2 * cluster_mean * moment_by_cluster[c][1] + cluster_mean^2 * n_values)
                              / cluster_variance)
                              + log(cluster_variance)*n_values
                            )

      end

      # Local update rule for "energy" for Potts distribution (Equation 6)
      log_probabilities[r] = number_of_neighbors_r * β + data_likelihood

    end

    max_log_probabilities = maximum(log_probabilities)

    # Normalize the probabilities and take the exponential to get the actual probabilities
    # .- does the vector-scalar subtraction and exp. does the element-wise exponential (instead of the matrix exponential)
    normalized_probabilities = exp.(log_probabilities .- max_log_probabilities)
    normalized_probabilities /= sum(normalized_probabilities)

    # Sample a new cluster assignment from the Multinomial with the computed probabilities
    state.w[n] = rand(Categorical(normalized_probabilities))

  end
  # println(time() - t0)
end
