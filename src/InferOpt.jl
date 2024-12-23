"""
    InferOpt

A toolbox for using combinatorial optimization algorithms within machine learning pipelines.

See our preprint <https://arxiv.org/abs/2207.13513>
"""
module InferOpt

using ChainRulesCore: ChainRulesCore, NoTangent, RuleConfig, Tangent, ZeroTangent
using ChainRulesCore: rrule, rrule_via_ad, unthunk
using DensityInterface: logdensityof
using DifferentiableExpectations:
    DifferentiableExpectations, Reinforce, empirical_predistribution, empirical_distribution
using Distributions:
    Distributions,
    ContinuousUnivariateDistribution,
    LogNormal,
    Normal,
    product_distribution,
    logpdf
using DocStringExtensions: TYPEDEF, TYPEDFIELDS, TYPEDSIGNATURES
using LinearAlgebra: dot
using Random: Random, AbstractRNG, GLOBAL_RNG, MersenneTwister, rand, seed!
using Statistics: mean
using StatsBase: StatsBase, sample
using StatsFuns: logaddexp, softmax
using RequiredInterfaces

include("interface.jl")

include("utils/utils.jl")
include("utils/some_functions.jl")
include("utils/pushforward.jl")
include("utils/linear_maximizer.jl")
include("utils/isotonic_regression/isotonic_l2.jl")
include("utils/isotonic_regression/isotonic_kl.jl")
include("utils/isotonic_regression/projection.jl")

# Layers
include("layers/simple/interpolation.jl")
include("layers/simple/identity.jl")

include("layers/perturbed/utils.jl")
include("layers/perturbed/perturbation.jl")
include("layers/perturbed/perturbed.jl")

include("layers/regularized/abstract_regularized.jl")
include("layers/regularized/soft_argmax.jl")
include("layers/regularized/sparse_argmax.jl")
include("layers/regularized/soft_rank.jl")
include("layers/regularized/regularized_frank_wolfe.jl")

if !isdefined(Base, :get_extension)
    include("../ext/InferOptFrankWolfeExt.jl")
end

# Losses
include("losses/fenchel_young_loss.jl")
include("losses/spoplus_loss.jl")
include("losses/ssvm_loss.jl")
include("losses/zero_one_loss.jl")
include("losses/imitation_loss.jl")

export half_square_norm
export shannon_entropy, negative_shannon_entropy
export one_hot_argmax, ranking
export LinearMaximizer, apply_g, apply_h, objective_value

export Pushforward

export IdentityRelaxation
export Interpolation

export AbstractRegularized
export SoftArgmax, soft_argmax
export SparseArgmax, sparse_argmax
export SoftRank, soft_rank, soft_rank_l2, soft_rank_kl
export SoftSort, soft_sort, soft_sort_l2, soft_sort_kl
export RegularizedFrankWolfe

export PerturbedOracle
export PerturbedAdditive
export PerturbedMultiplicative

export FenchelYoungLoss
export StructuredSVMLoss
export ImitationLoss, get_y_true
export SPOPlusLoss

end
