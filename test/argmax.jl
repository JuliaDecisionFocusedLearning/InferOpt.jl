using Flux
using InferOpt
using LinearAlgebra
using Random
using Test

Random.seed!(63)

# verbose = get(ENV, "CI", "false") == "false"
verbose = false

## Main functions

nb_features = 5
encoder_factory() = Chain(Dense(nb_features, 1), dropfirstdim, make_positive)
true_encoder = encoder_factory()
true_maximizer(θ; kwargs...) = one_hot_argmax(θ; kwargs...)
cost(y; instance) = dot(y, -true_encoder(instance))
error_function(ŷ, y) = hamming_distance(ŷ, y)

## Pipelines

pipelines_imitation_θ = [
    # SPO+
    (encoder=encoder_factory(), maximizer=identity, loss=SPOPlusLoss(true_maximizer)),
]

pipelines_imitation_y = [
    # Structured SVM
    (
        encoder=encoder_factory(),
        maximizer=identity,
        loss=StructuredSVMLoss(ZeroOneBaseLoss()),
    ),
    # Explicit regularized + FYL
    (encoder=encoder_factory(), maximizer=identity, loss=FenchelYoungLoss(sparse_argmax)),
    (encoder=encoder_factory(), maximizer=identity, loss=FenchelYoungLoss(soft_argmax)),
    # Explicit regularized + other loss
    (encoder=encoder_factory(), maximizer=sparse_argmax, loss=Flux.Losses.mse),
    (encoder=encoder_factory(), maximizer=soft_argmax, loss=Flux.Losses.mse),
    # Perturbed + FYL
    (
        encoder=encoder_factory(),
        maximizer=identity,
        loss=FenchelYoungLoss(PerturbedAdditive(true_maximizer; ε=1.0, nb_samples=5)),
    ),
    (
        encoder=encoder_factory(),
        maximizer=identity,
        loss=FenchelYoungLoss(PerturbedMultiplicative(true_maximizer; ε=1.0, nb_samples=5)),
    ),
    # Perturbed + other loss
    (
        encoder=encoder_factory(),
        maximizer=PerturbedAdditive(true_maximizer; ε=1.0, nb_samples=10),
        loss=Flux.Losses.mse,
    ),
    (
        encoder=encoder_factory(),
        maximizer=PerturbedMultiplicative(true_maximizer; ε=1.0, nb_samples=10),
        loss=Flux.Losses.mse,
    ),
    # Generic regularized + FYL
    (
        encoder=encoder_factory(),
        maximizer=identity,
        loss=FenchelYoungLoss(
            RegularizedGeneric(true_maximizer, half_square_norm, identity)
        ),
    ),
    # Generic regularized + other loss
    (
        encoder=encoder_factory(),
        maximizer=RegularizedGeneric(true_maximizer, half_square_norm, identity),
        loss=Flux.Losses.mse,
    ),
]

pipelines_experience = [
    (
        encoder=encoder_factory(),
        maximizer=identity,
        loss=Pushforward(PerturbedAdditive(true_maximizer; ε=1.0, nb_samples=10), cost),
    ),
    (
        encoder=encoder_factory(),
        maximizer=identity,
        loss=Pushforward(
            PerturbedMultiplicative(true_maximizer; ε=1.0, nb_samples=10), cost
        ),
    ),
    (
        encoder=encoder_factory(),
        maximizer=identity,
        loss=Pushforward(
            RegularizedGeneric(true_maximizer, half_square_norm, identity), cost
        ),
    ),
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
    pipeline_1 = deepcopy(pipeline)
    (; encoder, maximizer, loss) = pipeline_1
    pipeline_loss_imitation_θ(x, θ, y) = loss(maximizer(encoder(x)), θ)
    test_pipeline!(
        pipeline_1,
        pipeline_loss_imitation_θ;
        true_encoder=true_encoder,
        true_maximizer=true_maximizer,
        data_train=data_train,
        data_test=data_test,
        error_function=error_function,
        cost=cost,
        epochs=100,
        verbose=verbose,
        setting_name="argmax - imitation_θ",
    )

    pipeline_2 = deepcopy(pipeline)
    (; encoder, maximizer, loss) = pipeline_2
    pipeline_loss_imitation_θy(x, θ, y) = loss(maximizer(encoder(x)), θ, y)
    test_pipeline!(
        pipeline_2,
        pipeline_loss_imitation_θy;
        true_encoder=true_encoder,
        true_maximizer=true_maximizer,
        data_train=data_train,
        data_test=data_test,
        error_function=error_function,
        cost=cost,
        epochs=100,
        verbose=verbose,
        setting_name="argmax - imitation_θ - precomputed y_true",
    )
end

for pipeline in pipelines_imitation_y
    pipeline_1 = deepcopy(pipeline)
    (; encoder, maximizer, loss) = pipeline_1
    pipeline_loss_imitation_y(x, θ, y) = loss(maximizer(encoder(x)), y)
    test_pipeline!(
        pipeline_1,
        pipeline_loss_imitation_y;
        true_encoder=true_encoder,
        true_maximizer=true_maximizer,
        data_train=data_train,
        data_test=data_test,
        error_function=error_function,
        cost=cost,
        epochs=200,
        verbose=verbose,
        setting_name="argmax - imitation_y",
    )
end

for pipeline in pipelines_experience
    pipeline_1 = deepcopy(pipeline)
    (; encoder, maximizer, loss) = pipeline_1
    pipeline_loss_experience(x, θ, y) = loss(maximizer(encoder(x)); instance=x)
    test_pipeline!(
        pipeline_1,
        pipeline_loss_experience;
        true_encoder=true_encoder,
        true_maximizer=true_maximizer,
        data_train=data_train,
        data_test=data_test,
        error_function=error_function,
        cost=cost,
        epochs=500,
        verbose=verbose,
        setting_name="argmax - experience",
    )
end
