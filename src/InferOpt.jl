module InferOpt

using ChainRulesCore
using ForwardDiff
using LinearAlgebra
using Random
using SimpleTraits
using SparseArrays
using Statistics
using Test
using Unzip

include("interpolation/interpolation.jl")

include("regularized/proba.jl")
include("regularized/penalties.jl")
include("regularized/simplex.jl")
include("regularized/prediction.jl")
include("regularized/ranking.jl")

include("perturbed/abstract.jl")
include("perturbed/cost.jl")
include("perturbed/normal.jl")
include("perturbed/lognormal.jl")

include("fenchel_young/fenchel_young.jl")

include("smart_predict_optimize/smart_predict_optimize.jl")

include("structured_svm/structured_loss.jl")
include("structured_svm/structured_svm.jl")

include("utils/testing/Testing.jl")

export shannon_entropy, half_square_norm
export one_hot_argmax, softmax, sparsemax
export ranking
export IsRegularizedPrediction

export Interpolation

export AbstractPerturbed
export PerturbedNormal
export PerturbedLogNormal
export PerturbedCost

export FenchelYoungLoss

export SPOPlusLoss

export ZeroOneLoss, GeneralStructuredLoss
export IsStructuredLossFunction
export StructuredSVMLoss

end
