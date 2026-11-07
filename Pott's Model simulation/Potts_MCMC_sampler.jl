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

function update_μ_and_σ2!(state::MCMC_State, hyper_Parameters::Hyper_Parameters)
end
function update_z!(state::MCMC_State, hyper_parameters::Hyper_Parameters)
end

function update_w!(state::MCMC_State, hyper_parameters::Hyper_Parameters, data::Observed_Data)

  R = hyper_parameters.R
  N = data.N
  G = data.G
  β = hyper_parameters.β

  # Generate a permutation σ of J = {1,2,...,N}
  permutation = randperm(N)

  # For j in 1,...,N
  for n in permutation
    log_probabilities = zeros(R)
    
    for r in 1:R

      # Count the number of neighbors of spot n that are assigned to cluster r (δ(w_j, w_j') in Equation 6)
      number_of_neighbors_r = 0
      for neighbor in data.adjecency[n,:]
        if state.w[neighbor] == r && neighbor != 0
          number_of_neighbors_r += 1
        end
      end

      # Evaluation of the posterior distribution h_j (w_j) (Equation 12)
      data_likelihood = 0.0
      for g in 1:G
        gene_cluster = state.z[g]
        value = data.y[n,g]
        cluster_mean = state.μ[r,gene_cluster]
        cluster_variance = state.σ2[r,gene_cluster]

        data_likelihood += -0.5 * ((value-cluster_mean)^2 / cluster_variance) - 0.5 * log(cluster_variance)
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
end
