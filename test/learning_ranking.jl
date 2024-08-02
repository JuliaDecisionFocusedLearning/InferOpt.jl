@testitem "imit - SPO+ (θ)" default_imports = false begin
    include("InferOptTestUtils/src/InferOptTestUtils.jl")
    using InferOpt, .InferOptTestUtils, Random
    Random.seed!(63)

    test_pipeline!(
        PipelineLossImitationθ();
        instance_dim=5,
        true_maximizer=ranking,
        maximizer=identity_kw,
        loss=SPOPlusLoss(ranking),
        error_function=hamming_distance,
    )
end

@testitem "imit - SPO+ (θ & y)" default_imports = false begin
    include("InferOptTestUtils/src/InferOptTestUtils.jl")
    using InferOpt, .InferOptTestUtils, Random
    Random.seed!(63)

    test_pipeline!(
        PipelineLossImitationθy();
        instance_dim=5,
        true_maximizer=ranking,
        maximizer=identity_kw,
        loss=SPOPlusLoss(ranking),
        error_function=hamming_distance,
    )
end

@testitem "imit - MSE IdentityRelaxation" default_imports = false begin
    include("InferOptTestUtils/src/InferOptTestUtils.jl")
    using InferOpt, .InferOptTestUtils, LinearAlgebra, Random
    Random.seed!(63)

    test_pipeline!(
        PipelineLossImitation();
        instance_dim=5,
        true_maximizer=ranking,
        maximizer=normalize ∘ IdentityRelaxation(ranking),
        loss=mse_kw,
        error_function=hamming_distance,
    )
end

@testitem "imit - MSE Interpolation" default_imports = false begin
    include("InferOptTestUtils/src/InferOptTestUtils.jl")
    using InferOpt, .InferOptTestUtils, Random
    Random.seed!(63)

    test_pipeline!(
        PipelineLossImitation();
        instance_dim=5,
        true_maximizer=ranking,
        maximizer=Interpolation(ranking; λ=5.0),
        loss=mse_kw,
        error_function=hamming_distance,
    )
end

@testitem "imit - MSE PerturbedAdditive" default_imports = false begin
    include("InferOptTestUtils/src/InferOptTestUtils.jl")
    using InferOpt, .InferOptTestUtils, Random
    Random.seed!(63)

    test_pipeline!(
        PipelineLossImitation();
        instance_dim=5,
        true_maximizer=ranking,
        maximizer=PerturbedAdditive(ranking; ε=1.0, nb_samples=10),
        loss=mse_kw,
        error_function=hamming_distance,
    )
end

@testitem "imit - MSE PerturbedMultiplicative" default_imports = false begin
    include("InferOptTestUtils/src/InferOptTestUtils.jl")
    using InferOpt, .InferOptTestUtils, Random
    Random.seed!(63)

    test_pipeline!(
        PipelineLossImitation();
        instance_dim=5,
        true_maximizer=ranking,
        maximizer=PerturbedMultiplicative(ranking; ε=1.0, nb_samples=10),
        loss=mse_kw,
        error_function=hamming_distance,
    )
end

@testitem "imit - MSE RegularizedFrankWolfe" default_imports = false begin
    include("InferOptTestUtils/src/InferOptTestUtils.jl")
    using DifferentiableFrankWolfe, FrankWolfe, InferOpt, .InferOptTestUtils, Random
    Random.seed!(63)

    test_pipeline!(
        PipelineLossImitation();
        instance_dim=5,
        true_maximizer=ranking,
        maximizer=RegularizedFrankWolfe(
            ranking;
            Ω=half_square_norm,
            Ω_grad=identity_kw,
            frank_wolfe_kwargs=(; max_iteration=10, line_search=FrankWolfe.Agnostic()),
        ),
        loss=mse_kw,
        error_function=hamming_distance,
    )
end

@testitem "imit - FYL PerturbedAdditive" default_imports = false begin
    include("InferOptTestUtils/src/InferOptTestUtils.jl")
    using InferOpt, .InferOptTestUtils, Random
    Random.seed!(63)

    test_pipeline!(
        PipelineLossImitation();
        instance_dim=5,
        true_maximizer=ranking,
        maximizer=identity_kw,
        loss=FenchelYoungLoss(PerturbedAdditive(ranking; ε=1.0, nb_samples=5)),
        error_function=hamming_distance,
    )
end

@testitem "imit - FYL PerturbedMultiplicative" default_imports = false begin
    include("InferOptTestUtils/src/InferOptTestUtils.jl")
    using InferOpt, .InferOptTestUtils, Random
    Random.seed!(63)

    test_pipeline!(
        PipelineLossImitation();
        instance_dim=5,
        true_maximizer=ranking,
        maximizer=identity_kw,
        loss=FenchelYoungLoss(PerturbedMultiplicative(ranking; ε=1.0, nb_samples=5)),
        error_function=hamming_distance,
    )
end

@testitem "imit - FYL PerturbedAdditive{LogNormal}" default_imports = false begin
    include("InferOptTestUtils/src/InferOptTestUtils.jl")
    using InferOpt, .InferOptTestUtils, Random, Distributions, LinearAlgebra
    Random.seed!(63)

    test_pipeline!(
        PipelineLossImitation();
        instance_dim=5,
        true_maximizer=ranking,
        maximizer=identity_kw,
        loss=FenchelYoungLoss(
            PerturbedAdditive(
                ranking; ε=1.0, nb_samples=5, perturbation_dist=LogNormal(0, 1)
            ),
        ),
        error_function=hamming_distance,
    )
end

@testitem "imit - FYL RegularizedFrankWolfe" default_imports = false begin
    include("InferOptTestUtils/src/InferOptTestUtils.jl")
    using DifferentiableFrankWolfe, FrankWolfe, InferOpt, .InferOptTestUtils, Random
    Random.seed!(63)

    test_pipeline!(
        PipelineLossImitation();
        instance_dim=5,
        true_maximizer=ranking,
        maximizer=identity_kw,
        loss=FenchelYoungLoss(
            RegularizedFrankWolfe(
                ranking;
                Ω=half_square_norm,
                Ω_grad=identity_kw,
                frank_wolfe_kwargs=(; max_iteration=10, line_search=FrankWolfe.Agnostic()),
            ),
        ),
        error_function=hamming_distance,
    )
end

@testitem "exp - Pushforward PerturbedAdditive" default_imports = false begin
    include("InferOptTestUtils/src/InferOptTestUtils.jl")
    using InferOpt, .InferOptTestUtils, LinearAlgebra, Random, Statistics
    Random.seed!(63)

    true_encoder = encoder_factory()
    cost(y; instance) = dot(y, -true_encoder(instance))
    f(θ; kwargs...) = cost(ranking(θ; kwargs...); kwargs...)
    test_pipeline!(
        PipelineLossExperience();
        instance_dim=5,
        true_maximizer=ranking,
        maximizer=identity_kw,
        loss=PerturbedAdditive(f; ε=1.0, nb_samples=10),
        error_function=hamming_distance,
        true_encoder=true_encoder,
        cost=cost,
    )
end

@testitem "exp - Pushforward PerturbedMultiplicative" default_imports = false begin
    include("InferOptTestUtils/src/InferOptTestUtils.jl")
    using InferOpt, .InferOptTestUtils, LinearAlgebra, Random
    Random.seed!(63)

    true_encoder = encoder_factory()
    cost(y; instance) = dot(y, -true_encoder(instance))
    f(θ; kwargs...) = cost(ranking(θ; kwargs...); kwargs...)
    test_pipeline!(
        PipelineLossExperience();
        instance_dim=5,
        true_maximizer=ranking,
        maximizer=identity_kw,
        loss=PerturbedAdditive(f; ε=1.0, nb_samples=10),
        error_function=hamming_distance,
        true_encoder=true_encoder,
        cost=cost,
    )
end

@testitem "exp - Pushforward PerturbedAdditive{LogNormal}" default_imports = false begin
    include("InferOptTestUtils/src/InferOptTestUtils.jl")
    using InferOpt, .InferOptTestUtils, LinearAlgebra, Random, Distributions
    Random.seed!(63)

    true_encoder = encoder_factory()
    cost(y; instance) = dot(y, -true_encoder(instance))
    f(θ; kwargs...) = cost(ranking(θ; kwargs...); kwargs...)
    test_pipeline!(
        PipelineLossExperience();
        instance_dim=5,
        true_maximizer=ranking,
        maximizer=identity_kw,
        loss=PerturbedAdditive(f; ε=1.0, nb_samples=10, perturbation_dist=LogNormal(0, 1)),
        error_function=hamming_distance,
        true_encoder=true_encoder,
        cost=cost,
        epochs=100,
    )
end

@testitem "exp - Pushforward PerturbedMultiplicative{LogNormal}" default_imports = false begin
    include("InferOptTestUtils/src/InferOptTestUtils.jl")
    using InferOpt, .InferOptTestUtils, LinearAlgebra, Random, Distributions
    Random.seed!(63)

    true_encoder = encoder_factory()
    cost(y; instance) = dot(y, -true_encoder(instance))
    f(θ; kwargs...) = cost(ranking(θ; kwargs...); kwargs...)
    test_pipeline!(
        PipelineLossExperience();
        instance_dim=5,
        true_maximizer=ranking,
        maximizer=identity_kw,
        loss=PerturbedMultiplicative(
            f; ε=1.0, nb_samples=10, perturbation_dist=LogNormal(0, 1)
        ),
        error_function=hamming_distance,
        true_encoder=true_encoder,
        cost=cost,
        epochs=100,
    )
end

@testitem "exp - Pushforward PerturbedOracle{LogNormal}" default_imports = false begin
    include("InferOptTestUtils/src/InferOptTestUtils.jl")
    using InferOpt, .InferOptTestUtils, LinearAlgebra, Random, Distributions
    Random.seed!(63)

    p(θ) = MvLogNormal(θ, I)

    true_encoder = encoder_factory()
    cost(y; instance) = dot(y, -true_encoder(instance))
    test_pipeline!(
        PipelineLossExperience();
        instance_dim=5,
        true_maximizer=ranking,
        maximizer=identity_kw,
        loss=Pushforward(Perturbed(ranking, p; nb_samples=10), cost),
        error_function=hamming_distance,
        true_encoder=true_encoder,
        cost=cost,
    )
end

@testitem "exp - Pushforward RegularizedFrankWolfe" default_imports = false begin
    include("InferOptTestUtils/src/InferOptTestUtils.jl")
    using DifferentiableFrankWolfe,
        FrankWolfe, InferOpt, .InferOptTestUtils, LinearAlgebra, Random
    Random.seed!(63)

    true_encoder = encoder_factory()
    cost(y; instance) = dot(y, -true_encoder(instance))
    test_pipeline!(
        PipelineLossExperience();
        instance_dim=5,
        true_maximizer=ranking,
        maximizer=identity_kw,
        loss=Pushforward(
            RegularizedFrankWolfe(
                ranking;
                Ω=half_square_norm,
                Ω_grad=identity_kw,
                frank_wolfe_kwargs=(; max_iteration=10, line_search=FrankWolfe.Agnostic()),
            ),
            cost,
        ),
        error_function=hamming_distance,
        true_encoder=true_encoder,
        cost=cost,
    )
end

@testitem "exp - soft rank" default_imports = false begin
    include("InferOptTestUtils/src/InferOptTestUtils.jl")
    using InferOpt, .InferOptTestUtils, LinearAlgebra, Random, Test
    Random.seed!(63)

    true_encoder = encoder_factory()
    cost(y; instance) = dot(y, -true_encoder(instance))

    Random.seed!(67)
    soft_rank_l2_results = test_pipeline!(
        PipelineLossExperience();
        instance_dim=5,
        true_maximizer=ranking,
        maximizer=SoftRank(),
        loss=cost,
        error_function=hamming_distance,
        true_encoder=true_encoder,
        cost=cost,
        epochs=50,
    )

    Random.seed!(67)
    soft_rank_kl_results = test_pipeline!(
        PipelineLossExperience();
        instance_dim=5,
        true_maximizer=ranking,
        maximizer=SoftRank(; regularization="kl"),
        loss=cost,
        error_function=hamming_distance,
        true_encoder=true_encoder,
        cost=cost,
        epochs=50,
    )

    Random.seed!(67)
    perturbed_results = test_pipeline!(
        PipelineLossExperience();
        instance_dim=5,
        true_maximizer=ranking,
        maximizer=identity_kw,
        loss=Pushforward(PerturbedAdditive(ranking; ε=1.0, nb_samples=10), cost),
        error_function=hamming_distance,
        true_encoder=true_encoder,
        cost=cost,
        epochs=50,
    )

    # Check that we achieve better performance than the reinforce trick
    @test soft_rank_l2_results.test_cost_gaps[end] < perturbed_results.test_cost_gaps[end]
    @test soft_rank_kl_results.test_cost_gaps[end] < perturbed_results.test_cost_gaps[end]
end

@testitem "imit - FYL - soft rank" default_imports = false begin
    include("InferOptTestUtils/src/InferOptTestUtils.jl")
    using InferOpt, .InferOptTestUtils, LinearAlgebra, Random, Test
    Random.seed!(63)

    true_encoder = encoder_factory()

    test_pipeline!(
        PipelineLossImitation();
        instance_dim=5,
        true_maximizer=ranking,
        maximizer=identity_kw,
        loss=FenchelYoungLoss(SoftRank()),
        error_function=hamming_distance,
        true_encoder=true_encoder,
    )

    test_pipeline!(
        PipelineLossImitation();
        instance_dim=5,
        true_maximizer=ranking,
        maximizer=identity_kw,
        loss=FenchelYoungLoss(SoftRank(; regularization="kl", ε=10.0)),
        error_function=hamming_distance,
        true_encoder=true_encoder,
    )
end
