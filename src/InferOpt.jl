module InferOpt

using ChainRulesCore
using LinearAlgebra
using Random
using SimpleTraits
using SparseArrays
using Statistics
using Test

include("interpolation/interpolation.jl")

include("regularized/isregularized.jl")
include("regularized/regularized_utils.jl")
include("regularized/soft_argmax.jl")
include("regularized/sparse_argmax.jl")
include("regularized/frank_wolfe.jl")

include("perturbed/abstract_perturbed.jl")
include("perturbed/composition.jl")
include("perturbed/additive.jl")
include("perturbed/multiplicative.jl")

include("fenchel_young/fenchel_young.jl")

include("spo/spoplus_loss.jl")

include("ssvm/isbaseloss.jl")
include("ssvm/zeroone_baseloss.jl")
include("ssvm/ssvm_loss.jl")

export shannon_entropy, half_square_norm
export one_hot_argmax, ranking
export IsRegularized
export soft_argmax, sparse_argmax

export Interpolation

export AbstractPerturbed
export PerturbedAdditive
export PerturbedMultiplicative
export PerturbedComposition

export FenchelYoungLoss

export SPOPlusLoss

export IsBaseLoss
export ZeroOneBaseLoss
export StructuredSVMLoss

end
