"""
    InferOpt

A toolbox for using combinatorial optimization algorithms within machine learning pipelines.

See our preprint <https://arxiv.org/abs/2207.13513>
"""
module InferOpt

using ChainRulesCore: ChainRulesCore, NoTangent, RuleConfig, Tangent, ZeroTangent
using ChainRulesCore: rrule, rrule_via_ad, unthunk
using DensityInterface: logdensityof
using LinearAlgebra: dot
using Random: AbstractRNG, GLOBAL_RNG, MersenneTwister, rand, seed!
using Statistics: mean
using StatsBase: StatsBase, sample
using ThreadsX: ThreadsX

include("interface.jl")

include("utils/some_functions.jl")
include("utils/probability_distribution.jl")
include("utils/pushforward.jl")

include("simple/interpolation.jl")
include("simple/identity.jl")

include("regularized/abstract_regularized.jl")
include("regularized/soft_argmax.jl")
include("regularized/sparse_argmax.jl")
include("regularized/regularized_frank_wolfe.jl")

include("perturbed/abstract_perturbed.jl")
include("perturbed/additive.jl")
include("perturbed/multiplicative.jl")
include("perturbed/perturbed_oracle.jl")

include("imitation/spoplus_loss.jl")
include("imitation/ssvm_loss.jl")
include("imitation/fenchel_young_loss.jl")
include("imitation/imitation_loss.jl")
include("imitation/zero_one_loss.jl")

if !isdefined(Base, :get_extension)
    include("../ext/InferOptFrankWolfeExt.jl")
end

export half_square_norm
export shannon_entropy, negative_shannon_entropy
export one_hot_argmax, ranking

export FixedAtomsProbabilityDistribution
export compute_expectation
export compute_probability_distribution
export Pushforward

export IdentityRelaxation
export Interpolation

export AbstractRegularized
export SoftArgmax, soft_argmax
export SparseArgmax, sparse_argmax
export RegularizedFrankWolfe

export AbstractPerturbed
export PerturbedAdditive
export PerturbedMultiplicative
export PerturbedOracle

export FenchelYoungLoss
export StructuredSVMLoss
export ImitationLoss
export SPOPlusLoss

end
