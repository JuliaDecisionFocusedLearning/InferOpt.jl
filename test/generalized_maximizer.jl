using Random
using LinearAlgebra

Random.seed!(67)

CC = 10
function max_pricing(θ::AbstractVector; instance::AbstractMatrix)
    @assert length(θ) == size(instance, 1)
    @assert length(θ) == size(instance, 2)
    weights = θ .- instance
    return weights .>= 0
end

g(y; kwargs...) = vec(sum(y; dims=2))
h(y; instance) = -sum(dij * yij for (dij, yij) in zip(instance, y))

@testset "Generalized maximizer basics" begin
    instance = [
        1.0 2.0 0.0
        1.0 0.0 1.0
        3.0 4.0 2.0
    ]

    θ = [1.0, 0.0, 4.0]
    y = max_pricing(θ; instance)

    @test y == [1 0 1; 0 1 0; 1 1 1]

    generalized_maximizer = GeneralizedMaximizer(max_pricing, g, h)

    @test generalized_maximizer(θ; instance) == y

    val = InferOpt.objective_value(generalized_maximizer, θ, y; instance)

    @test val == θ' * g(y) + h(y; instance)
end

nb_features = 5
encoder_factory() = Chain(Dense(nb_features => 1; bias=false), dropfirstdim)#, make_positive)
true_encoder = encoder_factory()
generalized_maximizer = GeneralizedMaximizer(max_pricing, g, h)

data_train, data_test = generate_dataset(
    true_encoder,
    generalized_maximizer;
    nb_features=nb_features,
    instance_dim=5,
    nb_instances=100,
    noise_std=0.0,
);

function cost(y; instance)
    return -objective_value(generalized_maximizer, true_encoder(instance), y; instance)
end
error_function(ŷ, y) = hamming_distance(ŷ, y)

mse_loss(y1, y2; kwargs...) = Flux.Losses.mse(y1, y2)
identity_maximizer(θ; kwargs...) = identity(θ)

pipelines_imitation_y = [
    # Interpolation
    # (
    #     encoder=encoder_factory(),
    #     maximizer=Interpolation(generalized_maximizer; λ=5.0),
    #     loss=Flux.Losses.mse,
    # ),
    # Perturbed + FYL
    (
        encoder=encoder_factory(),
        maximizer=identity_maximizer,
        loss=FenchelYoungLoss(
            PerturbedAdditive(generalized_maximizer; ε=1.0, nb_samples=5)
        ),
    ),
    (
        encoder=encoder_factory(),
        maximizer=identity_maximizer,
        loss=FenchelYoungLoss(
            PerturbedMultiplicative(generalized_maximizer; ε=1.0, nb_samples=5)
        ),
    ),
    # Perturbed + other loss
    (
        encoder=encoder_factory(),
        maximizer=PerturbedAdditive(generalized_maximizer; ε=1.0, nb_samples=10),
        loss=mse_loss,
    ),
    (
        encoder=encoder_factory(),
        maximizer=PerturbedMultiplicative(generalized_maximizer; ε=1.0, nb_samples=50), # more samples were needed here
        loss=mse_loss,
    ),
    # # Generic regularized + FYL
    # (
    #     encoder=encoder_factory(),
    #     maximizer=identity,
    #     loss=FenchelYoungLoss(
    #         RegularizedGeneric(generalized_maximizer, half_square_norm, identity)
    #     ),
    # ),
    # # Generic regularized + other loss
    # (
    #     encoder=encoder_factory(),
    #     maximizer=RegularizedGeneric(generalized_maximizer, half_square_norm, identity),
    #     loss=Flux.Losses.mse,
    # ),
]

pipelines_imitation_θ = [
    # SPO+
    (
        encoder=encoder_factory(),
        maximizer=identity_maximizer,
        loss=SPOPlusLoss(generalized_maximizer),
    )
]

pipelines_experience = [
    (
        encoder=encoder_factory(),
        maximizer=identity_maximizer,
        loss=Pushforward(
            PerturbedAdditive(generalized_maximizer; ε=1.0, nb_samples=50), cost
        ),
    ),
    (
        encoder=encoder_factory(),
        maximizer=identity_maximizer,
        loss=Pushforward(
            PerturbedMultiplicative(generalized_maximizer; ε=1.0, nb_samples=50), cost
        ),
    ),
    (
        encoder=encoder_factory(),
        maximizer=identity_maximizer,
        loss=Pushforward(
            PerturbedAdditive(
                generalized_maximizer; ε=1.0, nb_samples=50, is_parallel=true
            ),
            cost,
        ),
    ),
    (
        encoder=encoder_factory(),
        maximizer=identity_maximizer,
        loss=Pushforward(
            PerturbedMultiplicative(
                generalized_maximizer; ε=1.0, nb_samples=50, is_parallel=true
            ),
            cost,
        ),
    ),
    # (
    #     encoder=encoder_factory(),
    #     maximizer=identity,
    #     loss=Pushforward(
    #         RegularizedGeneric(true_maximizer, half_square_norm, identity), cost
    #     ),
    # ),
]

for pipeline in pipelines_imitation_y
    pipeline = deepcopy(pipeline)
    (; encoder, maximizer, loss) = pipeline
    function pipeline_loss_imitation_y(x, θ, y)
        return loss(maximizer(encoder(x); instance=x), y; instance=x)
    end
    test_pipeline!(
        pipeline,
        pipeline_loss_imitation_y;
        true_encoder=true_encoder,
        true_maximizer=generalized_maximizer,
        data_train=data_train,
        data_test=data_test,
        error_function=error_function,
        cost=cost,
        epochs=500,
        verbose=true,
        setting_name="generalized maximizer - imitation_y",
    )
end

for pipeline in pipelines_imitation_θ
    pipeline_1 = deepcopy(pipeline)
    (; encoder, maximizer, loss) = pipeline_1
    function pipeline_loss_imitation_θ(x, θ, y)
        return loss(maximizer(encoder(x); instance=x), θ; instance=x)
    end
    test_pipeline!(
        pipeline_1,
        pipeline_loss_imitation_θ;
        true_encoder=true_encoder,
        true_maximizer=generalized_maximizer,
        data_train=data_train,
        data_test=data_test,
        error_function=error_function,
        cost=cost,
        epochs=100,
        verbose=true,
        setting_name="generalized maximizer - imitation_θ",
    )

    pipeline_2 = deepcopy(pipeline)
    (; encoder, maximizer, loss) = pipeline_2
    function pipeline_loss_imitation_θ(x, θ, y)
        return loss(maximizer(encoder(x); instance=x), θ, y; instance=x)
    end
    test_pipeline!(
        pipeline_2,
        pipeline_loss_imitation_θ;
        true_encoder=true_encoder,
        true_maximizer=generalized_maximizer,
        data_train=data_train,
        data_test=data_test,
        error_function=error_function,
        cost=cost,
        epochs=100,
        verbose=true,
        setting_name="generalized maximizer - imitation_θ - precomputed y_true",
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
        true_maximizer=generalized_maximizer,
        data_train=data_train,
        data_test=data_test,
        error_function=error_function,
        cost=cost,
        epochs=1000,
        verbose=true,
        setting_name="generalized maximizer - experience",
    )
end
