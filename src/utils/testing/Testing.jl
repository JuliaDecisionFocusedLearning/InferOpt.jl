module Testing

using ..InferOpt
using ..LinearAlgebra
using ..ProgressMeter
using ..Statistics
using ..Test
using ..UnicodePlots

include("dataset.jl")
include("trainer.jl")
include("metrics.jl")
include("error.jl")
include("loss.jl")
include("perf.jl")

dropfirstdim(z::AbstractArray) = dropdims(z; dims=1)

export generate_dataset
export mape, normalized_mape
export hamming_distance, normalized_hamming_distance
export define_flux_loss
export plot_perf, test_perf
export dropfirstdim
export train_test_split

export InferOptTrainer, InferOptModel, InferOptDataset
export Loss, HammingDistance, CostGap, ParameterError, MeanSquaredError
export compute_metrics!

end
