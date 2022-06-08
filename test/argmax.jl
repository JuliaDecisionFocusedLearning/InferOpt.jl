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
true_maximizer(θ; kwargs...) = one_hot_argmax(θ; kwargs...)
cost(y; instance) = dot(y, -true_encoder(instance))
error_function(ŷ, y) = hamming_distance(ŷ, y)

## Pipelines

pipelines = list_standard_pipelines(encoder_factory, true_maximizer; cost=cost)

append!(
    pipelines["y"],
    [
        # Structured SVM
        (
            encoder=encoder_factory(),
            maximizer=identity,
            loss=StructuredSVMLoss(ZeroOneLoss()),
        ),
        # Regularized prediction: explicit
        (encoder=encoder_factory(), maximizer=identity, loss=FenchelYoungLoss(sparsemax)),
        (
            encoder=encoder_factory(),
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
    setting_name="argmax",
)
