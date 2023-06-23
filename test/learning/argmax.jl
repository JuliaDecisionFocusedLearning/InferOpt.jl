include("../InferOptTestUtils/InferOptTestUtils.jl")
using FrankWolfe
using InferOpt
using .InferOptTestUtils
using Random
using Test

Random.seed!(63)

@testset "Argmax - imit - SPO+ (θ)" begin
    test_pipeline!(
        PipelineLossImitationθ;
        instance_dim=5,
        true_maximizer=one_hot_argmax,
        maximizer=identity,
        loss=SPOPlusLoss(one_hot_argmax),
        error_function=hamming_distance,
    )
end

@testset "Argmax - imit - SPO+ (θ & y)" begin
    test_pipeline!(
        PipelineLossImitationθy;
        instance_dim=5,
        true_maximizer=one_hot_argmax,
        maximizer=identity,
        loss=SPOPlusLoss(one_hot_argmax),
        error_function=hamming_distance,
    )
end

@testset "Argmax - imit - SSVM" begin
    test_pipeline!(
        PipelineLossImitation;
        instance_dim=5,
        true_maximizer=one_hot_argmax,
        maximizer=identity,
        loss=StructuredSVMLoss(ZeroOneBaseLoss()),
        error_function=hamming_distance,
    )
end

@testset "Argmax - imit - MSE sparse argmax" begin
    test_pipeline!(
        PipelineLossImitation;
        instance_dim=5,
        true_maximizer=one_hot_argmax,
        maximizer=sparse_argmax,
        loss=mse,
        error_function=hamming_distance,
    )
end

@testset "Argmax - imit - MSE soft argmax" begin
    test_pipeline!(
        PipelineLossImitation;
        instance_dim=5,
        true_maximizer=one_hot_argmax,
        maximizer=soft_argmax,
        loss=mse,
        error_function=hamming_distance,
    )
end

@testset "Argmax - imit - MSE PerturbedAdditive" begin
    test_pipeline!(
        PipelineLossImitation;
        instance_dim=5,
        true_maximizer=one_hot_argmax,
        maximizer=PerturbedAdditive(one_hot_argmax; ε=1.0, nb_samples=10),
        loss=mse,
        error_function=hamming_distance,
    )
end

@testset "Argmax - imit - MSE PerturbedMultiplicative" begin
    test_pipeline!(
        PipelineLossImitation;
        instance_dim=5,
        true_maximizer=one_hot_argmax,
        maximizer=PerturbedMultiplicative(one_hot_argmax; ε=1.0, nb_samples=10),
        loss=mse,
        error_function=hamming_distance,
    )
end

@testset "Argmax - imit - MSE RegularizedGeneric" begin
    test_pipeline!(
        PipelineLossImitation;
        instance_dim=5,
        true_maximizer=one_hot_argmax,
        maximizer=RegularizedGeneric(one_hot_argmax, half_square_norm, identity),
        loss=mse,
        error_function=hamming_distance,
        maximizer_kwargs=(
            frank_wolfe_kwargs=(line_search=FrankWolfe.Agnostic(), max_iteration=10),
        ),
    )
end

@testset "Argmax - imit - FYL sparse argmax" begin
    test_pipeline!(
        PipelineLossImitation;
        instance_dim=5,
        true_maximizer=one_hot_argmax,
        maximizer=identity,
        loss=FenchelYoungLoss(sparse_argmax),
        error_function=hamming_distance,
    )
end

@testset "Argmax - imit - FYL soft argmax" begin
    test_pipeline!(
        PipelineLossImitation;
        instance_dim=5,
        true_maximizer=one_hot_argmax,
        maximizer=identity,
        loss=FenchelYoungLoss(soft_argmax),
        error_function=hamming_distance,
    )
end

@testset "Argmax - imit - FYL PerturbedAdditive" begin
    test_pipeline!(
        PipelineLossImitation;
        instance_dim=5,
        true_maximizer=one_hot_argmax,
        maximizer=identity,
        loss=FenchelYoungLoss(PerturbedAdditive(one_hot_argmax; ε=1.0, nb_samples=5)),
        error_function=hamming_distance,
    )
end

@testset "Argmax - imit - FYL PerturbedMultiplicative" begin
    test_pipeline!(
        PipelineLossImitation;
        instance_dim=5,
        true_maximizer=one_hot_argmax,
        maximizer=identity,
        loss=FenchelYoungLoss(PerturbedMultiplicative(one_hot_argmax; ε=1.0, nb_samples=5)),
        error_function=hamming_distance,
    )
end

@testset "Argmax - imit - FYL RegularizedGeneric" begin
    test_pipeline!(
        PipelineLossImitation;
        instance_dim=5,
        true_maximizer=one_hot_argmax,
        maximizer=identity,
        loss=FenchelYoungLoss(
            RegularizedGeneric(one_hot_argmax, half_square_norm, identity)
        ),
        error_function=hamming_distance,
        loss_kwargs=(
            frank_wolfe_kwargs=(line_search=FrankWolfe.Agnostic(), max_iteration=10),
        ),
    )
end

@testset "Argmax - exp - Pushforward PerturbedAdditive" begin
    true_encoder = encoder_factory()
    cost(y; instance) = dot(y, -true_encoder(instance))
    test_pipeline!(
        PipelineLossExperience;
        instance_dim=5,
        true_maximizer=one_hot_argmax,
        maximizer=identity,
        loss=Pushforward(PerturbedAdditive(one_hot_argmax; ε=1.0, nb_samples=10), cost),
        error_function=hamming_distance,
        true_encoder=true_encoder,
        cost=cost,
    )
end

@testset "Argmax - exp - Pushforward PerturbedMultiplicative" begin
    true_encoder = encoder_factory()
    cost(y; instance) = dot(y, -true_encoder(instance))
    test_pipeline!(
        PipelineLossExperience;
        instance_dim=5,
        true_maximizer=one_hot_argmax,
        maximizer=identity,
        loss=Pushforward(
            PerturbedMultiplicative(one_hot_argmax; ε=1.0, nb_samples=10), cost
        ),
        error_function=hamming_distance,
        true_encoder=true_encoder,
        cost=cost,
    )
end

@testset "Argmax - exp - Pushforward RegularizedGeneric" begin
    true_encoder = encoder_factory()
    cost(y; instance, kwargs...) = dot(y, -true_encoder(instance))
    test_pipeline!(
        PipelineLossExperience;
        instance_dim=5,
        true_maximizer=one_hot_argmax,
        maximizer=identity,
        loss=Pushforward(
            RegularizedGeneric(one_hot_argmax, half_square_norm, identity), cost
        ),
        error_function=hamming_distance,
        true_encoder=true_encoder,
        cost=cost,
        loss_kwargs=(
            frank_wolfe_kwargs=(line_search=FrankWolfe.Agnostic(), max_iteration=10),
        ),
    )
end
