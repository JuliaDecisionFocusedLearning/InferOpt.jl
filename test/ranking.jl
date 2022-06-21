using Flux
using InferOpt
using InferOpt.Testing
using LinearAlgebra
using Random
using Test

Random.seed!(63)

## Main functions

nb_features = 5

encoder_factory() = Chain(Dense(nb_features, 1), dropfirstdim, make_positive)
true_encoder = encoder_factory()
true_maximizer(θ; kwargs...) = ranking(θ; kwargs...)
cost(y; instance) = dot(y, -true_encoder(instance))
error_function(ŷ, y) = hamming_distance(ŷ, y)

## Pipelines

pipelines = Dict{String,Vector}()

pipelines["θ"] = [(
    encoder=encoder_factory(), maximizer=identity, loss=SPOPlusLoss(true_maximizer)
)]

pipelines["(θ,y)"] = [(
    encoder=encoder_factory(), maximizer=identity, loss=SPOPlusLoss(true_maximizer)
)]

pipelines["y"] = [
    # Fenchel-Young loss (test forward pass)
    (
        encoder=encoder_factory(),
        maximizer=identity,
        loss=FenchelYoungLoss(PerturbedAdditive(true_maximizer; ε=1.0, nb_samples=3)),
    ),
    (
        encoder=encoder_factory(),
        maximizer=identity,
        loss=FenchelYoungLoss(PerturbedMultiplicative(true_maximizer; ε=0.5, nb_samples=3)),
    ),
    # Other differentiable loss (test backward pass)
    (
        encoder=encoder_factory(),
        maximizer=PerturbedAdditive(true_maximizer; ε=1.0, nb_samples=3),
        loss=Flux.Losses.mse,
    ),
    (
        encoder=encoder_factory(),
        maximizer=PerturbedMultiplicative(true_maximizer; ε=0.5, nb_samples=3),
        loss=Flux.Losses.mse,
    ),
    # Interpolation
    (
        encoder=encoder_factory(),
        maximizer=Interpolation(true_maximizer; λ=5.0),
        loss=Flux.Losses.mse,
    ),
]

pipelines["none"] = [
    (
        encoder=encoder_factory(),
        maximizer=identity,
        loss=cost ∘ PerturbedAdditive(true_maximizer; ε=1.0, nb_samples=3),
    ),
    (
        encoder=encoder_factory(),
        maximizer=identity,
        loss=cost ∘ PerturbedMultiplicative(true_maximizer; ε=0.5, nb_samples=3),
    ),
]

## Dataset generation

data_train, data_test = generate_dataset(
    true_encoder,
    true_maximizer;
    nb_features=nb_features,
    instance_dim=7,
    nb_instances=50,
    noise_std=0.01,
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
    verbose=true,
    setting_name="ranking",
)
