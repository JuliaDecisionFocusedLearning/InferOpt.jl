include("../InferOptTestUtils/InferOptTestUtils.jl")
using FrankWolfe
using InferOpt
using .InferOptTestUtils
using Random
using Test

Random.seed!(63)

@testset "Ranking - imit - SPO+ (θ)" begin
    test_pipeline!(
        PipelineLossImitationθ;
        instance_dim=5,
        true_maximizer=ranking,
        maximizer=identity,
        loss=SPOPlusLoss(ranking),
        error_function=hamming_distance,
    )
end

@testset "Ranking - imit - SPO+ (θ & y)" begin
    test_pipeline!(
        PipelineLossImitationθy;
        instance_dim=5,
        true_maximizer=ranking,
        maximizer=identity,
        loss=SPOPlusLoss(ranking),
        error_function=hamming_distance,
    )
end

@testset "Ranking - imit - MSE PlusIdentity" begin
    test_pipeline!(
        PipelineLossImitation;
        instance_dim=5,
        true_maximizer=ranking,
        maximizer=normalize ∘ PlusIdentity(ranking),
        loss=mse,
        error_function=hamming_distance,
    )
end

@testset "Ranking - imit - MSE Interpolation" begin
    test_pipeline!(
        PipelineLossImitation;
        instance_dim=5,
        true_maximizer=ranking,
        maximizer=Interpolation(ranking; λ=5.0),
        loss=mse,
        error_function=hamming_distance,
    )
end

@testset "Ranking - imit - MSE PerturbedAdditive" begin
    test_pipeline!(
        PipelineLossImitation;
        instance_dim=5,
        true_maximizer=ranking,
        maximizer=PerturbedAdditive(ranking; ε=1.0, nb_samples=10),
        loss=mse,
        error_function=hamming_distance,
    )
end

@testset "Ranking - imit - MSE PerturbedMultiplicative" begin
    test_pipeline!(
        PipelineLossImitation;
        instance_dim=5,
        true_maximizer=ranking,
        maximizer=PerturbedMultiplicative(ranking; ε=1.0, nb_samples=10),
        loss=mse,
        error_function=hamming_distance,
    )
end

@testset "Ranking - imit - MSE RegularizedGeneric" begin
    test_pipeline!(
        PipelineLossImitation;
        instance_dim=5,
        true_maximizer=ranking,
        maximizer=RegularizedGeneric(ranking, half_square_norm, identity),
        loss=mse,
        error_function=hamming_distance,
        maximizer_kwargs=(
            frank_wolfe_kwargs=(line_search=FrankWolfe.Agnostic(), max_iteration=10),
        ),
    )
end

@testset "Ranking - imit - FYL PerturbedAdditive" begin
    test_pipeline!(
        PipelineLossImitation;
        instance_dim=5,
        true_maximizer=ranking,
        maximizer=identity,
        loss=FenchelYoungLoss(PerturbedAdditive(ranking; ε=1.0, nb_samples=5)),
        error_function=hamming_distance,
    )
end

@testset "Ranking - imit - FYL PerturbedMultiplicative" begin
    test_pipeline!(
        PipelineLossImitation;
        instance_dim=5,
        true_maximizer=ranking,
        maximizer=identity,
        loss=FenchelYoungLoss(PerturbedMultiplicative(ranking; ε=1.0, nb_samples=5)),
        error_function=hamming_distance,
        epochs=100,
    )
end

@testset "Ranking - imit - FYL RegularizedGeneric" begin
    test_pipeline!(
        PipelineLossImitation;
        instance_dim=5,
        true_maximizer=ranking,
        maximizer=identity,
        loss=FenchelYoungLoss(RegularizedGeneric(ranking, half_square_norm, identity)),
        error_function=hamming_distance,
        epochs=100,
        loss_kwargs=(
            frank_wolfe_kwargs=(line_search=FrankWolfe.Agnostic(), max_iteration=10),
        ),
    )
end

@testset "Ranking - exp - Pushforward PerturbedAdditive" begin
    true_encoder = encoder_factory()
    cost(y; instance) = dot(y, -true_encoder(instance))
    test_pipeline!(
        PipelineLossExperience;
        instance_dim=5,
        true_maximizer=ranking,
        maximizer=identity,
        loss=Pushforward(PerturbedAdditive(ranking; ε=1.0, nb_samples=10), cost),
        error_function=hamming_distance,
        true_encoder=true_encoder,
        cost=cost,
        epochs=100,
    )
end

@testset "Ranking - exp - Pushforward PerturbedMultiplicative" begin
    true_encoder = encoder_factory()
    cost(y; instance) = dot(y, -true_encoder(instance))
    test_pipeline!(
        PipelineLossExperience;
        instance_dim=5,
        true_maximizer=ranking,
        maximizer=identity,
        loss=Pushforward(PerturbedMultiplicative(ranking; ε=1.0, nb_samples=10), cost),
        error_function=hamming_distance,
        true_encoder=true_encoder,
        cost=cost,
        epochs=100,
    )
end

@testset "Ranking - exp - Pushforward RegularizedGeneric" begin
    true_encoder = encoder_factory()
    cost(y; instance, kwargs...) = dot(y, -true_encoder(instance))
    test_pipeline!(
        PipelineLossExperience;
        instance_dim=5,
        true_maximizer=ranking,
        maximizer=identity,
        loss=Pushforward(RegularizedGeneric(ranking, half_square_norm, identity), cost),
        error_function=hamming_distance,
        true_encoder=true_encoder,
        cost=cost,
        epochs=100,
        loss_kwargs=(
            frank_wolfe_kwargs=(line_search=FrankWolfe.Agnostic(), max_iteration=10),
        ),
    )
end
