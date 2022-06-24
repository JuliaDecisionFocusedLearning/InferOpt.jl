using Flux
using InferOpt
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

pipelines_imitation_θ = [(
    encoder=encoder_factory(), maximizer=identity, loss=SPOPlusLoss(true_maximizer)
)]

pipelines_imitation_y = [
    # Fenchel-Young loss (test forward pass)
    (
        encoder=encoder_factory(),
        maximizer=identity,
        loss=FenchelYoungLoss(PerturbedAdditive(true_maximizer; ε=1.0, nb_samples=3)),
    ),
    (
        encoder=encoder_factory(),
        maximizer=identity,
        loss=FenchelYoungLoss(PerturbedMultiplicative(true_maximizer; ε=1.0, nb_samples=5)),
    ),
    # Other differentiable loss (test backward pass)
    (
        encoder=encoder_factory(),
        maximizer=PerturbedAdditive(true_maximizer; ε=1.0, nb_samples=3),
        loss=Flux.Losses.mse,
    ),
    (
        encoder=encoder_factory(),
        maximizer=PerturbedMultiplicative(true_maximizer; ε=1.0, nb_samples=5),
        loss=Flux.Losses.mse,
    ),
    # Interpolation
    (
        encoder=encoder_factory(),
        maximizer=Interpolation(true_maximizer; λ=5.0),
        loss=Flux.Losses.mse,
    ),
]

pipelines_experience = [
    (
        encoder=encoder_factory(),
        maximizer=identity,
        loss=cost ∘ PerturbedAdditive(true_maximizer; ε=1.0, nb_samples=3),
    ),
    (
        encoder=encoder_factory(),
        maximizer=identity,
        loss=cost ∘ PerturbedMultiplicative(true_maximizer; ε=1.0, nb_samples=5),
    ),
    (encoder=encoder_factory(), maximizer=Interpolation(true_maximizer; λ=5.0), loss=cost),
]

## Dataset generation

data_train, data_test = generate_dataset(
    true_encoder,
    true_maximizer;
    nb_features=nb_features,
    instance_dim=5,
    nb_instances=100,
    noise_std=0.01,
);

## Test loop

for pipeline in pipelines_imitation_θ
    pipeline = deepcopy(pipeline)
    (; encoder, maximizer, loss) = pipeline
    pipeline_loss_imitation_θ(x, θ, y) = loss(maximizer(encoder(x)), θ)
    test_pipeline!(
        pipeline,
        pipeline_loss_imitation_θ;
        true_encoder=true_encoder,
        true_maximizer=true_maximizer,
        data_train=data_train,
        data_test=data_test,
        error_function=error_function,
        cost=cost,
        epochs=100,
        verbose=true,
        setting_name="ranking - imitation_θ",
    )
end

for pipeline in pipelines_imitation_y
    pipeline = deepcopy(pipeline)
    (; encoder, maximizer, loss) = pipeline
    pipeline_loss_imitation_y(x, θ, y) = loss(maximizer(encoder(x)), y)
    test_pipeline!(
        pipeline,
        pipeline_loss_imitation_y;
        true_encoder=true_encoder,
        true_maximizer=true_maximizer,
        data_train=data_train,
        data_test=data_test,
        error_function=error_function,
        cost=cost,
        epochs=200,
        verbose=true,
        setting_name="ranking - imitation_y",
    )
end

for pipeline in pipelines_experience
    pipeline = deepcopy(pipeline)
    (; encoder, maximizer, loss) = pipeline
    pipeline_loss_experience(x, θ, y) = loss(maximizer(encoder(x)); instance=x)
    test_pipeline!(
        pipeline,
        pipeline_loss_experience;
        true_encoder=true_encoder,
        true_maximizer=true_maximizer,
        data_train=data_train,
        data_test=data_test,
        error_function=error_function,
        cost=cost,
        epochs=500,
        verbose=true,
        setting_name="ranking - experience",
    )
end
