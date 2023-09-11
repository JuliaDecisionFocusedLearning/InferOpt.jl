@testitem "Isotonic l2 compare to HiGHS" default_imports = false begin
    include("InferOptTestUtils/src/InferOptTestUtils.jl")
    using InferOpt, .InferOptTestUtils, Random, HiGHS, JuMP, Test
    Random.seed!(63)

    function isotonic_custom(y)
        model = Model(HiGHS.Optimizer)
        set_silent(model)

        @variable(model, x[1:length(y)])
        @objective(model, Min, sum((x[i] - y[i])^2 for i in eachindex(x)))
        @constraint(model, [i in 1:(length(x) - 1)], x[i] >= x[i + 1])

        optimize!(model)
        return value.(x)
    end

    for _ in 1:100
        y = randn(1000)
        x = isotonic_custom(y)
        x2 = InferOpt.isotonic_l2(y)
        @test all(isapprox.(x, x2, atol=1e-2))
    end
end

@testitem "Test jacobian against finite differences" default_imports = false begin
    include("InferOptTestUtils/src/InferOptTestUtils.jl")
    using InferOpt, .InferOptTestUtils, FiniteDifferences, Random, Test, Zygote
    Random.seed!(63)

    for _ in 1:100
        θ = randn(50)

        sort_jac = Zygote.jacobian(x -> soft_sort_l2(x; ε=10.0), θ)[1]
        sort_jac_fd = FiniteDifferences.jacobian(central_fdm(2, 1), x -> soft_sort_l2(x; ε=10.0), θ)[1]
        @test all(isapprox.(sort_jac, sort_jac_fd, atol=1e-4))

        sort_jac = Zygote.jacobian(x -> soft_sort_kl(x; ε=10.0), θ)[1]
        sort_jac_fd = FiniteDifferences.jacobian(central_fdm(2, 1), x -> soft_sort_kl(x; ε=10.0), θ)[1]
        @test all(isapprox.(sort_jac, sort_jac_fd, atol=1e-4))

        rank_jac = Zygote.jacobian(x -> soft_rank_l2(x; ε=10.0), θ)[1]
        rank_jac_fd = FiniteDifferences.jacobian(central_fdm(2, 1), x -> soft_rank_l2(x; ε=10.0), θ)[1]
        @test all(isapprox.(rank_jac, rank_jac_fd, atol=1e-4))

        rank_jac = Zygote.jacobian(x -> soft_rank_kl(x; ε=10.0), θ)[1]
        rank_jac_fd = FiniteDifferences.jacobian(central_fdm(2, 1), x -> soft_rank_kl(x; ε=10.0), θ)[1]
        @test all(isapprox.(rank_jac, rank_jac_fd, atol=1e-4))
    end
end

@testitem "Learn by experience soft rank l2" default_imports = false begin
    include("InferOptTestUtils/src/InferOptTestUtils.jl")
    using InferOpt, .InferOptTestUtils, LinearAlgebra, Random, Test
    Random.seed!(63)

    true_encoder = encoder_factory()
    cost(y; instance) = dot(y, -true_encoder(instance))
    test_pipeline!(
        PipelineLossExperience();
        instance_dim=5,
        true_maximizer=ranking,
        maximizer=soft_rank,
        loss=cost,
        error_function=hamming_distance,
        true_encoder=true_encoder,
        cost=cost,
    )
end

@testitem "Learn by experience soft rank kl" default_imports = false begin
    include("InferOptTestUtils/src/InferOptTestUtils.jl")
    using InferOpt, .InferOptTestUtils, LinearAlgebra, Random, Test
    Random.seed!(63)

    true_encoder = encoder_factory()
    cost(y; instance) = dot(y, -true_encoder(instance))
    test_pipeline!(
        PipelineLossExperience();
        instance_dim=5,
        true_maximizer=ranking,
        maximizer=soft_rank_kl,
        loss=cost,
        error_function=hamming_distance,
        true_encoder=true_encoder,
        cost=cost,
    )
end

@testitem "Fenchel-Young loss soft rank L2" default_imports = false begin
    include("InferOptTestUtils/src/InferOptTestUtils.jl")
    using InferOpt, .InferOptTestUtils, LinearAlgebra, Random, Test
    Random.seed!(63)

    true_encoder = encoder_factory()
    test_pipeline!(
        PipelineLossImitation();
        instance_dim=5,
        true_maximizer=ranking,
        maximizer=identity,
        loss=FenchelYoungLoss(SoftRank()),
        error_function=hamming_distance,
        true_encoder=true_encoder,
    )
end

@testitem "Fenchel-Young loss soft rank kl" default_imports = false begin
    include("InferOptTestUtils/src/InferOptTestUtils.jl")
    using InferOpt, .InferOptTestUtils, LinearAlgebra, Random, Test
    Random.seed!(63)

    true_encoder = encoder_factory()
    test_pipeline!(
        PipelineLossImitation();
        instance_dim=5,
        true_maximizer=ranking,
        maximizer=identity,
        loss=FenchelYoungLoss(SoftRank(; is_l2_regularized=false)),
        error_function=hamming_distance,
        true_encoder=true_encoder,
    )
end
