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
h(y; instance) = sum(dij * yij for (dij, yij) in zip(instance, y))

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

@testset "Training" begin
    @test true
end

nb_features = 5
encoder_factory() = Chain(Dense(nb_features => 1; bias=false), dropfirstdim, make_positive)
true_encoder = encoder_factory()
true_maximizer = GeneralizedMaximizer(max_pricing, g, h)
# max_pricing(θ; kwargs...)
cost(y; instance) = -objective_value(true_maximizer, true_encoder(instance), y; instance)
#cost(y; instance) = 0.0
error_function(ŷ, y) = hamming_distance(ŷ, y)

pipelines_imitation_y = [
    # Interpolation  # TODO: make it work
    # (
    #     encoder=encoder_factory(),
    #     maximizer=Interpolation(true_maximizer; λ=5.0),
    #     loss=Flux.Losses.mse,
    # ),
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


data_train, data_test = generate_dataset(
    true_encoder,
    true_maximizer;
    nb_features=nb_features,
    instance_dim=5,
    nb_instances=100,
    noise_std=0.0,
);

for pipeline in pipelines_imitation_y
    pipeline = deepcopy(pipeline)
    (; encoder, maximizer, loss) = pipeline
    pipeline_loss_imitation_y(x, θ, y) = loss(maximizer(encoder(x)), y; instance=x)
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
        setting_name="paths - imitation_y",
    )
end

# TODO: benchmark with usual workaround with g(y) directly as labels and output of true_maximizer
