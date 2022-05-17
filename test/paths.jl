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

true_encoder = Chain(Dense(nb_features, 1), dropfirstdim, z -> -exp.(z) .- 1.)
cost(y; instance) = dot(y, -true_encoder(instance))
error_function(y1, y2) = Flux.Losses.mse(y1, y2)

function true_maximizer(θ::AbstractMatrix{R}; kwargs...) where {R<:Real}
    g = GridGraph{Int,R}(-θ)
    path = grid_dijkstra(g, 1, nv(g))
    y = path_to_matrix(g, path)
    return y
end

## Pipelines

# pipelines = list_standard_pipelines(true_maximizer; cost=cost, nb_features=nb_features)

pipelines = Dict{String,Vector}()

pipelines["y"] = [
    # Perturbations
    # (
    #     encoder=Chain(Dense(nb_features, 1), dropfirstdim, z -> -exp.(z) .- 1.),
    #     maximizer=identity,
    #     loss=FenchelYoungLoss(Perturbed(true_maximizer; ε=0.1, M=5)),
    # ),
    (
        encoder=Chain(Dense(nb_features, 1), dropfirstdim, z -> -exp.(z) .- 1.),
        maximizer=identity,
        loss=FenchelYoungLoss(PerturbedLogNormal(true_maximizer; ε=0.1, M=5)),
    ),
]

## Dataset generation

data_train, data_test = generate_dataset(
    true_encoder,
    true_maximizer;
    nb_features=nb_features,
    instance_dim=(10, 15),
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
    epochs=100,
    show_plots=true,
    setting_name="paths",
)
