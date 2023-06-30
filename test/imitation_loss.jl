@testitem "ImitationLoss vs SSVM" default_imports = false begin
    include("InferOptTestUtils/InferOptTestUtils.jl")
    using InferOpt, .InferOptTestUtils, Random, Test
    Random.seed!(63)

    true_encoder = encoder_factory()

    Random.seed!(67)
    perf = test_pipeline!(
        PipelineLossImitationLoss;
        instance_dim=5,
        true_maximizer=one_hot_argmax,
        maximizer=identity,
        loss=ZeroOneImitationLoss(),
        error_function=hamming_distance,
        true_encoder,
    )

    Random.seed!(67)
    benchmark_perf = test_pipeline!(
        PipelineLossImitation;
        instance_dim=5,
        true_maximizer=one_hot_argmax,
        maximizer=identity,
        loss=ZeroOneStructuredSVMLoss(),
        error_function=hamming_distance,
        true_encoder,
        verbose=false,
    )

    # Both performances should be equivalent
    @test all(isapprox.(perf.train_losses, benchmark_perf.train_losses, rtol=0.001))
    @test all(isapprox.(perf.test_losses, benchmark_perf.test_losses, rtol=0.001))
end

@testitem "ImitationLoss vs FYL SparseMax" default_imports = false begin
    include("InferOptTestUtils/InferOptTestUtils.jl")
    using InferOpt, .InferOptTestUtils, Random, Test
    Random.seed!(63)

    true_encoder = encoder_factory()

    Random.seed!(67)
    perf = test_pipeline!(
        PipelineLossImitationLoss;
        instance_dim=5,
        true_maximizer=one_hot_argmax,
        maximizer=identity,
        loss=ImitationLoss(;
            δ=(y, t_true) -> 0,
            Ω=y -> half_square_norm(y),
            aux_loss_maximizer=(θ, t_true, α) -> sparse_argmax(θ),
        ),
        error_function=hamming_distance,
        true_encoder,
    )

    Random.seed!(67)
    benchmark_perf = test_pipeline!(
        PipelineLossImitation;
        instance_dim=5,
        true_maximizer=one_hot_argmax,
        maximizer=identity,
        loss=FenchelYoungLoss(SparseArgmax()),
        error_function=hamming_distance,
        true_encoder,
        verbose=false,
    )

    # Both performances should be equivalent
    @test all(isapprox.(perf.train_losses, benchmark_perf.train_losses, rtol=0.001))
    @test all(isapprox.(perf.test_losses, benchmark_perf.test_losses, rtol=0.001))
end

@testitem "ImitationLoss vs FYL SoftMax" default_imports = false begin
    include("InferOptTestUtils/InferOptTestUtils.jl")
    using InferOpt, .InferOptTestUtils, Random, Test
    Random.seed!(63)

    true_encoder = encoder_factory()

    Random.seed!(67)
    perf = test_pipeline!(
        PipelineLossImitationLoss;
        instance_dim=5,
        true_maximizer=one_hot_argmax,
        maximizer=identity,
        loss=ImitationLoss(;
            δ=(y, t_true) -> 0,
            Ω=y -> negative_shannon_entropy(y),
            aux_loss_maximizer=(θ, t_true, α) -> soft_argmax(θ),
        ),
        error_function=hamming_distance,
        true_encoder,
    )

    Random.seed!(67)
    benchmark_perf = test_pipeline!(
        PipelineLossImitation;
        instance_dim=5,
        true_maximizer=one_hot_argmax,
        maximizer=identity,
        loss=FenchelYoungLoss(SoftArgmax()),
        error_function=hamming_distance,
        true_encoder,
        verbose=false,
    )

    # Both performances should be equivalent
    @test all(isapprox.(perf.train_losses, benchmark_perf.train_losses, rtol=0.001))
    @test all(isapprox.(perf.test_losses, benchmark_perf.test_losses, rtol=0.001))
end

@testitem "ImitationLoss vs SPO+ (α = 1)" default_imports = false begin
    include("InferOptTestUtils/InferOptTestUtils.jl")
    using InferOpt, .InferOptTestUtils, LinearAlgebra, Random, Test
    Random.seed!(63)

    true_encoder = encoder_factory()

    function spo_predictor(θ, t_true, α; kwargs...)
        (; θ_true) = t_true
        θ_α = α * θ - θ_true
        y_α = one_hot_argmax(θ_α; kwargs...)
        return y_α
    end

    function spo_base_loss(y, t_true)
        (; θ_true, y_true) = t_true
        return dot(θ_true, y_true) - dot(θ_true, y)
    end

    Random.seed!(67)
    perf = test_pipeline!(
        PipelineLossImitationLoss;
        instance_dim=5,
        true_maximizer=one_hot_argmax,
        maximizer=identity,
        loss=ImitationLoss(;
            δ=spo_base_loss, Ω=y -> 0, α=1, aux_loss_maximizer=spo_predictor
        ),
        error_function=hamming_distance,
        true_encoder,
    )

    Random.seed!(67)
    benchmark_perf = test_pipeline!(
        PipelineLossImitationθy;
        instance_dim=5,
        true_maximizer=one_hot_argmax,
        maximizer=identity,
        loss=SPOPlusLoss(one_hot_argmax; α=1.0),
        error_function=hamming_distance,
        true_encoder,
        verbose=false,
    )

    # Both performances should be equivalent
    @test all(isapprox.(perf.train_losses, benchmark_perf.train_losses, rtol=0.001))
    @test all(isapprox.(perf.test_losses, benchmark_perf.test_losses, rtol=0.001))
end

@testitem "ImitationLoss vs SPO+ (α = 2)" default_imports = false begin
    include("InferOptTestUtils/InferOptTestUtils.jl")
    using InferOpt, .InferOptTestUtils, LinearAlgebra, Random, Test
    Random.seed!(63)

    true_encoder = encoder_factory()

    function spo_predictor(θ, t_true, α; kwargs...)
        (; θ_true) = t_true
        θ_α = 2 * θ - θ_true
        y_α = one_hot_argmax(θ_α; kwargs...)
        return y_α
    end

    function spo_base_loss(y, t_true)
        (; θ_true, y_true) = t_true
        return dot(θ_true, y_true) - dot(θ_true, y)
    end

    Random.seed!(67)
    perf = test_pipeline!(
        PipelineLossImitationLoss;
        instance_dim=5,
        true_maximizer=one_hot_argmax,
        maximizer=identity,
        loss=ImitationLoss(;
            δ=spo_base_loss, Ω=y -> 0, α=2, aux_loss_maximizer=spo_predictor
        ),
        error_function=hamming_distance,
        true_encoder,
    )

    Random.seed!(67)
    benchmark_perf = test_pipeline!(
        PipelineLossImitationθy;
        instance_dim=5,
        true_maximizer=one_hot_argmax,
        maximizer=identity,
        loss=SPOPlusLoss(one_hot_argmax),
        error_function=hamming_distance,
        true_encoder,
        verbose=false,
    )

    # Both performances should be equivalent
    @test all(isapprox.(perf.train_losses, benchmark_perf.train_losses, rtol=0.001))
    @test all(isapprox.(perf.test_losses, benchmark_perf.test_losses, rtol=0.001))
end
