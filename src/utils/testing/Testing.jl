module Testing

using InferOpt
using LinearAlgebra
using Statistics
using Test

include("dataset.jl")
include("error.jl")
include("perf.jl")
include("pipelines.jl")

export generate_dataset
export mape, normalized_mape
export hamming_distance, normalized_hamming_distance
export define_pipeline_loss
export init_perf, update_perf!, plot_perf, test_perf

end
