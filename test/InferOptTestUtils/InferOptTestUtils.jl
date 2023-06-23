module InferOptTestUtils

using Graphs
using GridGraphs
using LinearAlgebra
using Lux
using Optimisers
using Statistics
using Test
using UnicodePlots

VERBOSE = get(ENV, "CI", "false") == "false"

dropfirstdim(z::AbstractArray) = dropdims(z; dims=1)
make_positive(z::AbstractArray) = log1p.(exp.(z))

const NB_FEATURES = 5
const NB_INSTANCES = 100
const NOISE_STD = 0.01
const EPOCHS = 100
const DECREASE = 0.75
const ENCODER = Chain(Dense(NB_FEATURES, 1), dropfirstdim, make_positive)

include("const.jl")
include("maximizers.jl")
include("dataset.jl")
include("error.jl")
include("perf.jl")
include("loss.jl")
include("pipeline.jl")

export DECREASE, ENCODER, EPOCHS, NB_FEATURES, NB_INSTANCES, NOISE_STD, VERBOSE
export shortest_path_maximizer
export encoder_factory, generate_dataset
export mse, mape, normalized_mape, hamming_distance, normalized_hamming_distance
export init_perf, update_perf!
export dropfirstdim, make_positive
export PipelineLossImitation, PipelineLossImitationθ, PipelineLossImitationθy
export PipelineLossExperience
export PipelineLossImitationLoss
export test_pipeline!

end
