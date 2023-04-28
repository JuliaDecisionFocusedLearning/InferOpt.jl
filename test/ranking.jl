@testitem "Ranking - imit - SPO+ (θ)" default_imports = false begin
    include("InferOptTestUtils/InferOptTestUtils.jl")
    using InferOpt, .InferOptTestUtils, Random
    Random.seed!(63)

    test_pipeline!(
        PipelineLossImitationθ;
        instance_dim=5,
        true_maximizer=ranking,
        maximizer=identity,
        loss=SPOPlusLoss(ranking),
        error_function=hamming_distance,
    )
end

@testitem "Ranking - imit - SPO+ (θ & y)" default_imports = false begin
    include("InferOptTestUtils/InferOptTestUtils.jl")
    using InferOpt, .InferOptTestUtils, Random
    Random.seed!(63)

    test_pipeline!(
        PipelineLossImitationθy;
        instance_dim=5,
        true_maximizer=ranking,
        maximizer=identity,
        loss=SPOPlusLoss(ranking),
        error_function=hamming_distance,
    )
end

@testitem "Ranking - imit - MSE PlusIdentity" default_imports = false begin
    include("InferOptTestUtils/InferOptTestUtils.jl")
    using InferOpt, .InferOptTestUtils, LinearAlgebra, Random
    Random.seed!(63)

    test_pipeline!(
        PipelineLossImitation;
        instance_dim=5,
        true_maximizer=ranking,
        maximizer=normalize ∘ PlusIdentity(ranking),
        loss=mse,
        error_function=hamming_distance,
    )
end

@testitem "Ranking - imit - MSE Interpolation" default_imports = false begin
    include("InferOptTestUtils/InferOptTestUtils.jl")
    using InferOpt, .InferOptTestUtils, Random
    Random.seed!(63)

    test_pipeline!(
        PipelineLossImitation;
        instance_dim=5,
        true_maximizer=ranking,
        maximizer=Interpolation(ranking; λ=5.0),
        loss=mse,
        error_function=hamming_distance,
    )
end

@testitem "Ranking - imit - MSE PerturbedAdditive" default_imports = false begin
    include("InferOptTestUtils/InferOptTestUtils.jl")
    using InferOpt, .InferOptTestUtils, Random
    Random.seed!(63)

    test_pipeline!(
        PipelineLossImitation;
        instance_dim=5,
        true_maximizer=ranking,
        maximizer=PerturbedAdditive(ranking; ε=1.0, nb_samples=10),
        loss=mse,
        error_function=hamming_distance,
    )
end

@testitem "Ranking - imit - MSE PerturbedMultiplicative" default_imports = false begin
    include("InferOptTestUtils/InferOptTestUtils.jl")
    using InferOpt, .InferOptTestUtils, Random
    Random.seed!(63)

    test_pipeline!(
        PipelineLossImitation;
        instance_dim=5,
        true_maximizer=ranking,
        maximizer=PerturbedMultiplicative(ranking; ε=1.0, nb_samples=10),
        loss=mse,
        error_function=hamming_distance,
    )
end

@testitem "Ranking - imit - MSE RegularizedGeneric" default_imports = false begin
    include("InferOptTestUtils/InferOptTestUtils.jl")
    using InferOpt, .InferOptTestUtils, Random
    Random.seed!(63)

    test_pipeline!(
        PipelineLossImitation;
        instance_dim=5,
        true_maximizer=ranking,
        maximizer=RegularizedGeneric(ranking, half_square_norm, identity),
        loss=mse,
        error_function=hamming_distance,
    )
end

@testitem "Ranking - imit - FYL PerturbedAdditive" default_imports = false begin
    include("InferOptTestUtils/InferOptTestUtils.jl")
    using InferOpt, .InferOptTestUtils, Random
    Random.seed!(63)

    test_pipeline!(
        PipelineLossImitation;
        instance_dim=5,
        true_maximizer=ranking,
        maximizer=identity,
        loss=FenchelYoungLoss(PerturbedAdditive(ranking; ε=1.0, nb_samples=5)),
        error_function=hamming_distance,
    )
end

@testitem "Ranking - imit - FYL PerturbedMultiplicative" default_imports = false begin
    include("InferOptTestUtils/InferOptTestUtils.jl")
    using InferOpt, .InferOptTestUtils, Random
    Random.seed!(63)

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

@testitem "Ranking - imit - FYL RegularizedGeneric" default_imports = false begin
    include("InferOptTestUtils/InferOptTestUtils.jl")
    using InferOpt, .InferOptTestUtils, Random
    Random.seed!(63)

    test_pipeline!(
        PipelineLossImitation;
        instance_dim=5,
        true_maximizer=ranking,
        maximizer=identity,
        loss=FenchelYoungLoss(RegularizedGeneric(ranking, half_square_norm, identity)),
        error_function=hamming_distance,
        epochs=100,
    )
end

@testitem "Ranking - exp - Pushforward PerturbedAdditive" default_imports = false begin
    include("InferOptTestUtils/InferOptTestUtils.jl")
    using InferOpt, .InferOptTestUtils, LinearAlgebra, Random
    Random.seed!(63)

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

@testitem "Ranking - exp - Pushforward PerturbedMultiplicative" default_imports = false begin
    include("InferOptTestUtils/InferOptTestUtils.jl")
    using InferOpt, .InferOptTestUtils, LinearAlgebra, Random
    Random.seed!(63)

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

@testitem "Ranking - exp - Pushforward RegularizedGeneric" default_imports = false begin
    include("InferOptTestUtils/InferOptTestUtils.jl")
    using InferOpt, .InferOptTestUtils, LinearAlgebra, Random
    Random.seed!(63)

    true_encoder = encoder_factory()
    cost(y; instance) = dot(y, -true_encoder(instance))
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
    )
end
