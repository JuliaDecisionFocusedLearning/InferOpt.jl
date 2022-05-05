module InferOpt

using ChainRulesCore
using DataStructures
using Distributions
using Folds
using Graphs
using LinearAlgebra
using Random
using SimpleTraits
using SparseArrays
using Statistics
using Test
using UnicodePlots

include("utils/proba.jl")
include("utils/error.jl")
include("utils/testing.jl")

include("regularized/penalties.jl")
include("regularized/simplex.jl")
include("regularized/prediction.jl")
include("regularized/ranking.jl")

include("interpolation.jl")
include("perturbed.jl")
include("fenchel_young.jl")
include("smart_predict_optimize.jl")
include("structured_svm.jl")

include("grid_graphs/abstract.jl")
include("grid_graphs/acyclic.jl")
include("grid_graphs/symmetric.jl")
include("grid_graphs/shortest_paths.jl")

export shannon_entropy, half_square_norm
export one_hot_argmax, softmax, sparsemax
export ranking
export IsRegularizedPrediction

export Interpolation
export Perturbed, PerturbedCost
export PerturbedGeneric
export FenchelYoungLoss
export SPOPlusLoss
export ZeroOneLoss, GeneralStructuredLoss
export IsStructuredLossFunction
export StructuredSVMLoss

export AcyclicGridGraph, SymmetricGridGraph
export grid_shortest_path, grid_shortest_path_cost

end
