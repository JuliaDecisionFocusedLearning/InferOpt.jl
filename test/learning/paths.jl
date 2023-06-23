include("../InferOptTestUtils/InferOptTestUtils.jl")
using FrankWolfe
using InferOpt
using .InferOptTestUtils
using Random
using Test

Random.seed!(63)

@testset "Paths - imit - SPO+ (θ)" begin
    test_pipeline!(
        PipelineLossImitationθ;
        instance_dim=(5, 5),
        true_maximizer=shortest_path_maximizer,
        maximizer=identity,
        loss=SPOPlusLoss(shortest_path_maximizer),
        error_function=mse,
    )
end

@testset "Paths - imit - SPO+ (θ & y)" begin
    test_pipeline!(
        PipelineLossImitationθy;
        instance_dim=(5, 5),
        true_maximizer=shortest_path_maximizer,
        maximizer=identity,
        loss=SPOPlusLoss(shortest_path_maximizer),
        error_function=mse,
    )
end

@testset "Paths - imit - MSE PlusIdentity" begin
    test_pipeline!(
        PipelineLossImitation;
        instance_dim=(5, 5),
        true_maximizer=shortest_path_maximizer,
        maximizer=normalize ∘ PlusIdentity(shortest_path_maximizer),
        loss=mse,
        error_function=mse,
    )
end

# @testset "Paths - imit - MSE Interpolation" begin
#     test_pipeline!(
#         PipelineLossImitation;
#         instance_dim=(5, 5),
#         true_maximizer=shortest_path_maximizer,
#         maximizer=Interpolation(shortest_path_maximizer; λ=5.0),
#         loss=mse,
#         error_function=mse,
#     )
# end  # TODO: make it work (doesn't seem to depend on λ)

@testset "Paths - imit - MSE PerturbedAdditive" begin
    test_pipeline!(
        PipelineLossImitation;
        instance_dim=(5, 5),
        true_maximizer=shortest_path_maximizer,
        maximizer=PerturbedAdditive(shortest_path_maximizer; ε=1.0, nb_samples=10),
        loss=mse,
        error_function=mse,
    )
end

@testset "Paths - imit - MSE PerturbedMultiplicative" begin
    test_pipeline!(
        PipelineLossImitation;
        instance_dim=(5, 5),
        true_maximizer=shortest_path_maximizer,
        maximizer=PerturbedMultiplicative(shortest_path_maximizer; ε=1.0, nb_samples=10),
        loss=mse,
        error_function=mse,
    )
end

@testset "Paths - imit - MSE RegularizedGeneric" begin
    test_pipeline!(
        PipelineLossImitation;
        instance_dim=(5, 5),
        true_maximizer=shortest_path_maximizer,
        maximizer=RegularizedGeneric(shortest_path_maximizer, half_square_norm, identity),
        loss=mse,
        error_function=mse,
        maximizer_kwargs=(
            frank_wolfe_kwargs=(line_search=FrankWolfe.Agnostic(), max_iteration=10),
        ),
    )
end

@testset "Paths - imit - FYL PerturbedAdditive" begin
    test_pipeline!(
        PipelineLossImitation;
        instance_dim=(5, 5),
        true_maximizer=shortest_path_maximizer,
        maximizer=identity,
        loss=FenchelYoungLoss(
            PerturbedAdditive(shortest_path_maximizer; ε=1.0, nb_samples=5)
        ),
        error_function=mse,
    )
end

@testset "Paths - imit - FYL PerturbedMultiplicative" begin
    test_pipeline!(
        PipelineLossImitation;
        instance_dim=(5, 5),
        true_maximizer=shortest_path_maximizer,
        maximizer=identity,
        loss=FenchelYoungLoss(
            PerturbedMultiplicative(shortest_path_maximizer; ε=1.0, nb_samples=5)
        ),
        error_function=mse,
        epochs=100,
    )
end

@testset "Paths - imit - FYL RegularizedGeneric" begin
    test_pipeline!(
        PipelineLossImitation;
        instance_dim=(5, 5),
        true_maximizer=shortest_path_maximizer,
        maximizer=identity,
        loss=FenchelYoungLoss(
            RegularizedGeneric(shortest_path_maximizer, half_square_norm, identity)
        ),
        error_function=mse,
        epochs=100,
        loss_kwargs=(
            frank_wolfe_kwargs=(line_search=FrankWolfe.Agnostic(), max_iteration=10),
        ),
    )
end

@testset "Paths - exp - Pushforward PerturbedAdditive" begin
    true_encoder = encoder_factory()
    cost(y; instance) = dot(y, -true_encoder(instance))
    test_pipeline!(
        PipelineLossExperience;
        instance_dim=(5, 5),
        true_maximizer=shortest_path_maximizer,
        maximizer=identity,
        loss=Pushforward(
            PerturbedAdditive(shortest_path_maximizer; ε=1.0, nb_samples=10), cost
        ),
        error_function=mse,
        true_encoder=true_encoder,
        cost=cost,
        epochs=500,
    )
end

@testset "Paths - exp - Pushforward PerturbedMultiplicative" begin
    true_encoder = encoder_factory()
    cost(y; instance) = dot(y, -true_encoder(instance))
    test_pipeline!(
        PipelineLossExperience;
        instance_dim=(5, 5),
        true_maximizer=shortest_path_maximizer,
        maximizer=identity,
        loss=Pushforward(
            PerturbedMultiplicative(shortest_path_maximizer; ε=1.0, nb_samples=10), cost
        ),
        error_function=mse,
        true_encoder=true_encoder,
        cost=cost,
        epochs=500,
    )
end

@testset "Paths - exp - Pushforward RegularizedGeneric" begin
    true_encoder = encoder_factory()
    cost(y; instance, kwargs...) = dot(y, -true_encoder(instance))
    test_pipeline!(
        PipelineLossExperience;
        instance_dim=(5, 5),
        true_maximizer=shortest_path_maximizer,
        maximizer=identity,
        loss=Pushforward(
            RegularizedGeneric(shortest_path_maximizer, half_square_norm, identity), cost
        ),
        error_function=mse,
        true_encoder=true_encoder,
        cost=cost,
        epochs=200,
        loss_kwargs=(
            frank_wolfe_kwargs=(line_search=FrankWolfe.Agnostic(), max_iteration=10),
        ),
    )
end
