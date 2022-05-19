using Flux
using Graphs
using GridGraphs
using InferOpt
using InferOpt.Testing
using LinearAlgebra
using Random
using Test

## Main functions

nb_features = 5

encoder_factory() = Chain(Dense(nb_features, 1), dropfirstdim)
true_encoder = encoder_factory()
cost(y; instance) = dot(y, -true_encoder(instance))

function true_maximizer(θ::AbstractMatrix{R}; kwargs...) where {R<:Real}
    g = AcyclicGridGraph{Int,R}(-θ)
    path = grid_topological_sort(g, 1, nv(g))
    y = path_to_matrix(g, path)
    return y
end

## Pipelines

pipelines = list_standard_pipelines(encoder_factory, true_maximizer; cost=cost)

## Dataset generation

data = generate_dataset(
    true_encoder,
    true_maximizer;
    nb_features=nb_features,
    instance_dim=(10, 12),
    nb_instances=200,
    noise_std=0.01,
);

## Test loop
metrics = Dict(
    "loss" => Loss,
    "mse" => MeanSquaredError,
    "cost gap" => CostGap,
    "parameter error" => ParameterError
)
test_loop(
    pipelines, data, true_maximizer, cost, true_encoder, metrics;
    nb_epochs=500, show_plots=true, setting_name="paths"
)
