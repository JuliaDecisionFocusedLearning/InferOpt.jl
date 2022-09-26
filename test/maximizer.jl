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

@testset "Training" begin
    @test true
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

# x, θ, y = data_train[1][1], data_train[2][1], data_train[3][1]
# x
# θ
# y
# cost(y; instance=x)
# cost(zero(y); instance=x)
# generalized_maximizer(θ; instance=x)

# θ
# dot(θ, generalized_maximizer.g(y; instance=x)) .+ generalized_maximizer.h(y; instance=x)

mse_loss(y1, y2; kwargs...) = Flux.Losses.mse(y1, y2)
identity_maximizer(θ; kwargs...) = identity(θ)

pipelines_imitation_y = [
    # Interpolation  # TODO: make it work
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

# data_train2 = (data_train[1], data_train[2], g.(data_train[3]));
# data_test2 = (data_test[1], data_test[2], g.(data_test[3]));

# maximizer2(θ; instance) = g(max_pricing(θ; instance))

# pipeline = (
#     encoder=encoder_factory(),
#     maximizer=identity,
#     loss=FenchelYoungLoss(PerturbedAdditive(maximizer2; ε=1.0, nb_samples=5)),
# )

# pipeline = deepcopy(pipeline)
# (; encoder, maximizer, loss) = pipeline
# pipeline_loss_imitation_y(x, θ, y) = loss(maximizer(encoder(x)), y; instance=x)
# test_pipeline!(
#     pipeline,
#     pipeline_loss_imitation_y;
#     true_encoder=true_encoder,
#     generalized_maximizer=maximizer2,
#     data_train=data_train2,
#     data_test=data_test2,
#     error_function=error_function,
#     cost=cost,
#     epochs=200,
#     verbose=true,
#     setting_name="paths - imitation_y",
# )

encoder = encoder_factory()
maximizer = identity_maximizer
loss = FenchelYoungLoss(PerturbedAdditive(generalized_maximizer; ε=1.0, nb_samples=10))

x = data_train[1][1]
θ = data_train[2][1]
y = data_train[3][1]

loss(θ, y; instance=x)

history = Float64[]
for e in -100.0:0.1:100.0
    θ[2] = e
    value = loss(θ, y; instance=x)
    push!(history, value)
end

println(lineplot(history))
