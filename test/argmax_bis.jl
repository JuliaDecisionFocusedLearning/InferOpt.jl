using Flux
using InferOpt
using InferOpt.Testing
using LinearAlgebra
using Random
using Test

Random.seed!(63)

include("utils.jl")

## Main functions

nb_features = 5

true_encoder = Chain(Dense(nb_features, 1), dropfirstdim)
true_maximizer(θ; kwargs...) = one_hot_argmax(θ; kwargs...)
cost(y; instance) = dot(y, -true_encoder(instance))

## Pipelines

pipelines = list_standard_pipelines(true_maximizer; cost=cost, nb_features=nb_features)

append!(
    pipelines["y"],
    [
        # Structured SVM
        (
            encoder=Chain(Dense(nb_features, 1), dropfirstdim),
            maximizer=identity,
            loss=StructuredSVMLoss(ZeroOneLoss()),
        ),
        # Regularized prediction: explicit
        (
            encoder=Chain(Dense(nb_features, 1), dropfirstdim),
            maximizer=identity,
            loss=FenchelYoungLoss(one_hot_argmax),
        ),
        (
            encoder=Chain(Dense(nb_features, 1), dropfirstdim),
            maximizer=identity,
            loss=FenchelYoungLoss(sparsemax),
        ),
        (
            encoder=Chain(Dense(nb_features, 1), dropfirstdim),
            maximizer=identity,
            loss=FenchelYoungLoss(InferOpt.softmax),
        ),
    ],
);

## Dataset generation

data_train, data_test = generate_dataset(
    true_encoder,
    true_maximizer;
    nb_features=nb_features,
    instance_dim=10,
    nb_instances=100,
    noise_std=0.02,
);

X_train, θ_train, Y_train = data_train
X_test, θ_test, Y_test = data_test
data = InferOptDataset(X_train, X_test, θ_train, θ_test, Y_train, Y_test)

## Test loop
test_loop(pipelines, data, true_maximizer; nb_epochs=500, show_plots=true)
