module Testing

using InferOpt
using LinearAlgebra
using Statistics
using Test

include("dataset.jl")
include("error.jl")
include("pipelines.jl")
include("perf.jl")

dropfirstdim(z::AbstractArray) = dropdims(z; dims=1)
make_negative(z::AbstractArray; threshold=0.) = -exp.(z) - threshold

export generate_dataset
export mape, normalized_mape
export hamming_distance, normalized_hamming_distance
export define_pipeline_loss
export list_standard_pipelines
export init_perf, update_perf!, plot_perf, test_perf
export dropfirstdim, make_negative

end
