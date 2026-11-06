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
function update_w!(state::MCMC_State, hyper_parameters::Hyper_Parameters)
end
