using Flux
using InferOpt
using InferOpt.Testing
using LinearAlgebra
using Random
using Test

## Main functions

nb_features = 5

encoder_factory() = Chain(Dense(nb_features, 1), dropfirstdim)
true_encoder = encoder_factory()
true_maximizer(θ; kwargs...) = ranking(θ; kwargs...)
cost(y; instance) = dot(y, -true_encoder(instance))

## Pipelines

pipelines = list_standard_pipelines(encoder_factory, true_maximizer; cost=cost)

push!(
    pipelines["y"],
    (
        encoder=encoder_factory(),
        maximizer=Interpolation(true_maximizer; λ=10.0),
        loss=Flux.Losses.mse,
    ),
);

## Dataset generation

data = generate_dataset(
    true_encoder,
    true_maximizer;
    nb_features=nb_features,
    instance_dim=10,
    nb_instances=100,
    noise_std=0.02,
);

## Test loop
metrics = Dict(
    "loss" => Loss,
    "hamming distance" => HammingDistance,
    "cost gap" => CostGap,
    "parameter error" => ParameterError
)
test_loop(
    pipelines, data, true_maximizer, cost, true_encoder, metrics;
    nb_epochs=500, show_plots=true, setting_name="ranking"
)
