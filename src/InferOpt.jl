module InferOpt

using ChainRulesCore: ChainRulesCore, NoTangent, RuleConfig, Tangent, ZeroTangent
using ChainRulesCore: rrule, rrule_via_ad, unthunk
using LinearAlgebra: dot
using Random: AbstractRNG, GLOBAL_RNG, MersenneTwister, rand, seed!
using SimpleTraits: SimpleTraits
using SimpleTraits: @traitdef, @traitfn, @traitimpl
using Statistics: mean
using StatsBase: StatsBase, sample
using ThreadsX: ThreadsX

include("utils/probability_distribution.jl")
include("utils/pushforward.jl")

include("plus_identity/plus_identity.jl")

include("interpolation/interpolation.jl")

include("regularized/regularized_utils.jl")
include("regularized/soft_argmax.jl")
include("regularized/sparse_argmax.jl")
include("regularized/regularized.jl")
include("regularized/frank_wolfe_optimizer.jl")

include("perturbed/abstract_perturbed.jl")
include("perturbed/additive.jl")
include("perturbed/multiplicative.jl")

include("fenchel_young/perturbed.jl")
include("fenchel_young/fenchel_young.jl")

include("spo/spoplus_loss.jl")

include("ssvm/isbaseloss.jl")
include("ssvm/zeroone_baseloss.jl")
include("ssvm/ssvm_loss.jl")

include("imitation_loss/imitation_loss.jl")

if !isdefined(Base, :get_extension)
    include("../ext/InferOptFrankWolfeExt.jl")
end

export FixedAtomsProbabilityDistribution
export compute_expectation, compress_distribution!
export Pushforward
export compute_probability_distribution

export PlusIdentity

export Interpolation

export half_square_norm
export shannon_entropy, negative_shannon_entropy
export one_hot_argmax, ranking
export soft_argmax, sparse_argmax
export Regularized

export PerturbedAdditive
export PerturbedMultiplicative

export FenchelYoungLoss

export SPOPlusLoss

export IsBaseLoss
export ZeroOneBaseLoss
export StructuredSVMLoss

export ImitationLoss, get_y_true

export SparseArgmax, SoftArgmax

export RegularizedFrankWolfe

end
