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

true_encoder = Chain(Dense(nb_features, 1), dropfirstdim)
cost(y; instance) = dot(y, -true_encoder(instance))
error_function(y1, y2) = Flux.Losses.mse(y1, y2)

function true_maximizer(θ::AbstractMatrix{R}; kwargs...) where {R<:Real}
    g = AcyclicGridGraph{Int,R}(-θ)
    shortest_path_tree = GridGraphs.grid_topological_sort(g, 1)
    path = GridGraphs.get_path(shortest_path_tree, 1, nv(g))
    y = GridGraphs.path_to_matrix(g, path)
    return y
end

## Pipelines

pipelines = list_standard_pipelines(true_maximizer; cost=cost, nb_features=nb_features)

## Dataset generation

data_train, data_test = generate_dataset(
    true_encoder,
    true_maximizer;
    nb_features=nb_features,
    instance_dim=(10, 20),
    nb_instances=100,
    noise_std=0.02,
);

## Test loop

test_loop(
    pipelines;
    true_encoder=true_encoder,
    true_maximizer=true_maximizer,
    data_train=data_train,
    data_test=data_test,
    error_function=error_function,
    cost=cost,
    epochs=500,
    show_plots=true,
    setting_name="paths",
)
