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

include("perturbed/perturbed_abstract.jl")
include("perturbed/perturbed_normal.jl")

include("fenchel_young/fenchel_young.jl")

include("smart_predict_optimize/smart_predict_optimize.jl")

include("structured_svm/structured_loss.jl")
include("structured_svm/structured_svm.jl")

include("experimental/perturbed_lognormal.jl")

include("utils/testing/Testing.jl")

export shannon_entropy, half_square_norm
export one_hot_argmax, softmax, sparsemax
export ranking
export IsRegularizedPrediction

export Interpolation

export Perturbed, PerturbedCost

export FenchelYoungLoss

export SPOPlusLoss

export ZeroOneLoss, GeneralStructuredLoss
export IsStructuredLossFunction
export StructuredSVMLoss

end
