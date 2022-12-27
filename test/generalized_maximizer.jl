using Random
using LinearAlgebra

Random.seed!(67)

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
    # (
    #     encoder=encoder_factory(),
    #     maximizer=PerturbedAdditive(generalized_maximizer; ε=1.0, nb_samples=10),
    #     loss=mse_loss,
    # ),
    # (
    #     encoder=encoder_factory(),
    #     maximizer=PerturbedMultiplicative(generalized_maximizer; ε=1.0, nb_samples=10),
    #     loss=mse_loss,
    # ),
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
        epochs=200,
        verbose=true,
        setting_name="paths - imitation_y",
    )
end
