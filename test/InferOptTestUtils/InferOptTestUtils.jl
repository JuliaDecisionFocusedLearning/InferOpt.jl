module InferOptTestUtils

using Flux
using Flux.Losses
using LinearAlgebra
using ProgressMeter
using Statistics
using Test
using UnicodePlots

include("const.jl")
include("dataset.jl")
include("error.jl")
include("perf.jl")
include("pipeline.jl")

export DECREASE, EPOCHS, NB_FEATURES, NB_INSTANCES, NOISE_STD
export generate_dataset, generate_predictions
export mse, mape, normalized_mape, hamming_distance, normalized_hamming_distance
export init_perf, update_perf!
export dropfirstdim, make_positive
export test_pipeline!

end
