@testitem "Generalized maximizer basics" default_imports = false begin
    include("InferOptTestUtils/src/InferOptTestUtils.jl")
    using InferOpt, .InferOptTestUtils, Test

    instance = [
        1.0 2.0 0.0
        1.0 0.0 1.0
        3.0 4.0 2.0
    ]

    θ = [1.0, 0.0, 4.0]
    y = max_pricing(θ; instance)

    @test y == [1 0 1; 0 1 0; 1 1 1]

    generalized_maximizer = LinearMaximizer(max_pricing; g, h)

    @test generalized_maximizer(θ; instance) == y

    val = InferOpt.objective_value(generalized_maximizer, θ, y; instance)

    @test val == θ' * g(y; instance) + h(y; instance)
end

@testitem "Generalized maximizer - imit - MSE PerturbedAdditive" default_imports = false begin
    include("InferOptTestUtils/src/InferOptTestUtils.jl")
    using InferOpt, .InferOptTestUtils, Random
    Random.seed!(63)

    true_encoder = encoder_factory()

    maximizer = LinearMaximizer(max_pricing; g, h)
    perturbed = PerturbedAdditive(maximizer; ε=1.0, nb_samples=10)
    function cost(y; instance)
        return -objective_value(maximizer, true_encoder(instance), y; instance)
    end

    test_pipeline!(
        PipelineLossImitation();
        instance_dim=5,
        true_maximizer=max_pricing,
        maximizer=perturbed,
        loss=mse_kw,
        error_function=hamming_distance,
        cost,
        true_encoder,
    )
end

@testitem "Generalized maximizer - imit - MSE PerturbedMultiplicative" default_imports =
    false begin
    include("InferOptTestUtils/src/InferOptTestUtils.jl")
    using InferOpt, .InferOptTestUtils, Random
    Random.seed!(63)

    true_encoder = encoder_factory()

    maximizer = LinearMaximizer(max_pricing; g, h)
    perturbed = PerturbedMultiplicative(maximizer; ε=1.0, nb_samples=10)
    function cost(y; instance)
        return -objective_value(maximizer, true_encoder(instance), y; instance)
    end

    test_pipeline!(
        PipelineLossImitation();
        instance_dim=5,
        true_maximizer=max_pricing,
        maximizer=perturbed,
        loss=mse_kw,
        error_function=hamming_distance,
        cost,
        true_encoder,
    )
end

@testitem "Generalized maximizer - imit - FYL PerturbedAdditive" default_imports = false begin
    include("InferOptTestUtils/src/InferOptTestUtils.jl")
    using InferOpt, .InferOptTestUtils, Random
    Random.seed!(63)

    true_encoder = encoder_factory()

    maximizer = LinearMaximizer(max_pricing; g, h)
    @info maximizer g h
    perturbed = PerturbedAdditive(maximizer; ε=1.0, nb_samples=10)
    @info perturbed
    function cost(y; instance)
        return -objective_value(maximizer, true_encoder(instance), y; instance)
    end

    test_pipeline!(
        PipelineLossImitation();
        instance_dim=5,
        true_maximizer=max_pricing,
        maximizer=identity_kw,
        loss=FenchelYoungLoss(perturbed),
        error_function=hamming_distance,
        cost,
        true_encoder,
    )
end

@testitem "Generalized maximizer - imit - FYL PerturbedMultiplicative" default_imports =
    false begin
    include("InferOptTestUtils/src/InferOptTestUtils.jl")
    using InferOpt, .InferOptTestUtils, Random
    Random.seed!(63)

    true_encoder = encoder_factory()

    maximizer = LinearMaximizer(max_pricing; g, h)
    perturbed = PerturbedMultiplicative(maximizer; ε=0.1, nb_samples=10)
    function cost(y; instance)
        return -objective_value(maximizer, true_encoder(instance), y; instance)
    end

    test_pipeline!(
        PipelineLossImitation();
        instance_dim=5,
        true_maximizer=max_pricing,
        maximizer=identity_kw,
        loss=FenchelYoungLoss(perturbed),
        error_function=hamming_distance,
        cost,
        true_encoder,
    )
end

@testitem "Generalized maximizer - imit - SPO+ (θ)" default_imports = false begin
    include("InferOptTestUtils/src/InferOptTestUtils.jl")
    using InferOpt, .InferOptTestUtils, Random
    Random.seed!(63)

    true_encoder = encoder_factory()

    generalized_maximizer = LinearMaximizer(max_pricing; g, h)
    function cost(y; instance)
        return -objective_value(generalized_maximizer, true_encoder(instance), y; instance)
    end

    test_pipeline!(
        PipelineLossImitationθ();
        instance_dim=5,
        true_maximizer=max_pricing,
        maximizer=identity_kw,
        loss=SPOPlusLoss(generalized_maximizer),
        error_function=hamming_distance,
        cost,
        true_encoder,
    )
end

@testitem "Generalized maximizer - imit - SPO+ (θ & y)" default_imports = false begin
    include("InferOptTestUtils/src/InferOptTestUtils.jl")
    using InferOpt, .InferOptTestUtils, Random
    Random.seed!(63)

    true_encoder = encoder_factory()

    generalized_maximizer = LinearMaximizer(max_pricing; g, h)
    function cost(y; instance)
        return -objective_value(generalized_maximizer, true_encoder(instance), y; instance)
    end

    test_pipeline!(
        PipelineLossImitationθy();
        instance_dim=5,
        true_maximizer=max_pricing,
        maximizer=identity_kw,
        loss=SPOPlusLoss(generalized_maximizer),
        error_function=hamming_distance,
        cost,
        true_encoder,
    )
end

@testitem "Generalized maximizer - exp - Pushforward PerturbedAdditive" default_imports =
    false begin
    include("InferOptTestUtils/src/InferOptTestUtils.jl")
    using InferOpt, .InferOptTestUtils, LinearAlgebra, Random
    Random.seed!(63)

    true_encoder = encoder_factory()

    generalized_maximizer = LinearMaximizer(max_pricing; g, h)
    function cost(y; instance)
        return -objective_value(generalized_maximizer, true_encoder(instance), y; instance)
    end

    test_pipeline!(
        PipelineLossExperience();
        instance_dim=5,
        true_maximizer=max_pricing,
        maximizer=identity_kw,
        loss=Pushforward(
            PerturbedAdditive(generalized_maximizer; ε=1.0, nb_samples=10), cost
        ),
        error_function=hamming_distance,
        true_encoder,
        cost,
    )
end

@testitem "Generalized maximizer - exp - Pushforward PerturbedMultiplicative" default_imports =
    false begin
    include("InferOptTestUtils/src/InferOptTestUtils.jl")
    using InferOpt, .InferOptTestUtils, LinearAlgebra, Random
    Random.seed!(63)

    true_encoder = encoder_factory()

    generalized_maximizer = LinearMaximizer(max_pricing; g, h)
    function cost(y; instance)
        return -objective_value(generalized_maximizer, true_encoder(instance), y; instance)
    end

    test_pipeline!(
        PipelineLossExperience();
        instance_dim=5,
        true_maximizer=max_pricing,
        maximizer=identity_kw,
        loss=Pushforward(
            PerturbedMultiplicative(generalized_maximizer; ε=1.0, nb_samples=10), cost
        ),
        error_function=hamming_distance,
        true_encoder,
        cost,
    )
end

@testitem "Regularized with a GeneralizedMaximizer" default_imports = false begin
    include("InferOptTestUtils/src/InferOptTestUtils.jl")
    using InferOpt, .InferOptTestUtils, Random, RequiredInterfaces, Test
    const RI = RequiredInterfaces
    Random.seed!(63)

    struct MyRegularized{M<:LinearMaximizer} <: AbstractRegularized # GeneralizedMaximizer
        maximizer::M
    end

    (regularized::MyRegularized)(θ; kwargs...) = regularized.maximizer(θ; kwargs...)
    function InferOpt.compute_regularization(regularized::MyRegularized, y)
        return InferOpt.sparse_argmax_regularization(y)
    end
    InferOpt.get_maximizer(regularized::MyRegularized) = regularized.maximizer

    @test RI.check_interface_implemented(AbstractRegularized, MyRegularized)

    regularized = MyRegularized(LinearMaximizer(sparse_argmax))

    test_pipeline!(
        PipelineLossImitation();
        instance_dim=5,
        true_maximizer=one_hot_argmax,
        maximizer=identity_kw,
        loss=FenchelYoungLoss(regularized),
        error_function=hamming_distance,
    )
end
