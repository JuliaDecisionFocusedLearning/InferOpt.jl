module InferOpt

using ChainRulesCore
using Random
using SimpleTraits
using SparseArrays
using Statistics
using UnPack

include("utils.jl")
include("interpolation.jl")
include("regularized.jl")
include("perturbed.jl")
include("fenchel_young.jl")
include("smart_predict_optimize.jl")
include("structured_svm.jl")

export one_hot_argmax, softmax, sparsemax
export shannon_entropy, half_square_norm
export Interpolation
export Perturbed, PerturbedCost
export FenchelYoungLoss
export SPOPlusLoss
export ZeroOneLoss, GeneralStructuredLoss
export StructuredSVMLoss

end
