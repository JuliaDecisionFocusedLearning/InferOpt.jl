module Testing

using ..InferOpt
using ..LinearAlgebra
using ..ProgressMeter
using ..Statistics
using ..Test
using ..UnicodePlots

include("dataset.jl")
include("error.jl")
include("loss.jl")
include("perf.jl")

dropfirstdim(z::AbstractArray) = dropdims(z; dims=1)

export generate_dataset
export mape, normalized_mape
export hamming_distance, normalized_hamming_distance
export define_flux_loss
export init_perf, update_perf!, plot_perf, test_perf
export dropfirstdim

end
