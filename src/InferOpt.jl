module InferOpt

using ChainRulesCore
using FrankWolfe: FrankWolfe
using FrankWolfe: ActiveSet, Adaptive, LinearMinimizationOracle
using FrankWolfe: away_frank_wolfe, compute_extreme_point
using Krylov: gmres
using LinearAlgebra
using LinearOperators: LinearOperator
using Random
using SimpleTraits: SimpleTraits
using SimpleTraits: @traitdef, @traitfn, @traitimpl
using SparseArrays
using Statistics
using Test

include("interpolation/interpolation.jl")

include("frank_wolfe/frank_wolfe_utils.jl")
include("frank_wolfe/differentiable_frank_wolfe.jl")

include("regularized/isregularized.jl")
include("regularized/regularized_utils.jl")
include("regularized/soft_argmax.jl")
include("regularized/sparse_argmax.jl")
include("regularized/regularized_generic.jl")

include("perturbed/abstract_perturbed.jl")
include("perturbed/composition.jl")
include("perturbed/additive.jl")
include("perturbed/multiplicative.jl")

include("fenchel_young/fenchel_young.jl")

include("spo/spoplus_loss.jl")

include("ssvm/isbaseloss.jl")
include("ssvm/zeroone_baseloss.jl")
include("ssvm/ssvm_loss.jl")

export Interpolation

export DifferentiableFrankWolfe

export shannon_entropy, half_square_norm
export one_hot_argmax, ranking
export IsRegularized
export soft_argmax, sparse_argmax
export RegularizedGeneric

export PerturbedAdditive
export PerturbedMultiplicative
export PerturbedComposition

export FenchelYoungLoss

export SPOPlusLoss

export IsBaseLoss
export ZeroOneBaseLoss
export StructuredSVMLoss

end
