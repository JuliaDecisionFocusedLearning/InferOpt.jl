module InferOpt

using ChainRulesCore
using LinearAlgebra
using Random
using SimpleTraits
using SparseArrays
using Statistics
using Test

include("interpolation/interpolation.jl")

include("regularized/proba.jl")
include("regularized/penalties.jl")
include("regularized/simplex.jl")
include("regularized/prediction.jl")
include("regularized/ranking.jl")

include("perturbed/abstract.jl")
include("perturbed/additive.jl")
include("perturbed/multiplicative.jl")

include("fenchel_young/fenchel_young.jl")

include("smart_predict_optimize/smart_predict_optimize.jl")

include("structured_svm/structured_loss.jl")
include("structured_svm/structured_svm.jl")

include("utils/testing/Testing.jl")

export shannon_entropy, half_square_norm
export one_hot_argmax, soft_argmax, sparse_argmax
export ranking
export IsRegularizedPrediction

export Interpolation

export AbstractPerturbed
export PerturbedAdditive
export PerturbedMultiplicative
export PerturbedComposition

export FenchelYoungLoss

export SPOPlusLoss

export ZeroOneLoss, GeneralStructuredLoss
export IsStructuredLossFunction
export StructuredSVMLoss

end
