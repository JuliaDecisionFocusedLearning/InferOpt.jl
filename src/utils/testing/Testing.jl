module Testing

using InferOpt
using LinearAlgebra
using Logging
using Statistics
using Test

include("dataset.jl")
include("trainer.jl")
include("metrics.jl")
include("error.jl")
include("pipelines.jl")
include("perf.jl")

dropfirstdim(z::AbstractArray) = dropdims(z; dims=1)
make_negative(z::AbstractArray; threshold=0.) = -exp.(z) - threshold

export generate_dataset
export mape, normalized_mape
export hamming_distance, normalized_hamming_distance
export define_pipeline_loss
export plot_perf, test_perf
export dropfirstdim, make_negative
export train_test_split
export list_standard_pipelines

export InferOptTrainer, InferOptDataset, AbstractScalarMetric
export compute_metrics!
export Loss, HammingDistance, CostGap, ParameterError, MeanSquaredError

end
