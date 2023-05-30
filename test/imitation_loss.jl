@testitem "ImitationLoss vs SSVM" default_imports = false begin
#! format: noindent
include("InferOptTestUtils/InferOptTestUtils.jl")
using InferOpt, .InferOptTestUtils, Random, Test
Random.seed!(63)

true_encoder = encoder_factory()

ssvm_base_loss = ZeroOneBaseLoss()

Random.seed!(67)
perf = test_pipeline!(
    PipelineLossImitationLoss;
    instance_dim=5,
    true_maximizer=one_hot_argmax,
    maximizer=identity,
    loss=ImitationLoss(
        (θ, t_true) ->
            InferOpt.compute_maximizer(ssvm_base_loss, θ, 1.0, get_y_true(t_true));
        base_loss=(y, t_true) -> ssvm_base_loss(y, t_true.y_true),
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
    loss=StructuredSVMLoss(ZeroOneBaseLoss()),
    error_function=hamming_distance,
    true_encoder,
    verbose=false,
)

# Both performances should be equivalent
@test all(isapprox.(perf.train_losses, benchmark_perf.train_losses, rtol=0.001))
@test all(isapprox.(perf.test_losses, benchmark_perf.test_losses, rtol=0.001))
end

@testitem "ImitationLoss vs FYL sparsemax" default_imports = false begin
#! format: noindent
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
    loss=ImitationLoss((θ, t_true) -> sparse_argmax(θ); Ω=half_square_norm),
    error_function=hamming_distance,
    true_encoder,
)

Random.seed!(67)
benchmark_perf = test_pipeline!(
    PipelineLossImitation;
    instance_dim=5,
    true_maximizer=one_hot_argmax,
    maximizer=identity,
    loss=FenchelYoungLoss(sparse_argmax),
    error_function=hamming_distance,
    true_encoder,
    verbose=false,
)

# Both performances should be equivalent
@test all(isapprox.(perf.train_losses, benchmark_perf.train_losses, rtol=0.001))
@test all(isapprox.(perf.test_losses, benchmark_perf.test_losses, rtol=0.001))
end

@testitem "ImitationLoss vs FYL softmax" default_imports = false begin
#! format: noindent
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
    loss=ImitationLoss((θ, t_true) -> soft_argmax(θ); Ω=negative_shannon_entropy),
    error_function=hamming_distance,
    true_encoder,
)

Random.seed!(67)
benchmark_perf = test_pipeline!(
    PipelineLossImitation;
    instance_dim=5,
    true_maximizer=one_hot_argmax,
    maximizer=identity,
    loss=FenchelYoungLoss(soft_argmax),
    error_function=hamming_distance,
    true_encoder,
    verbose=false,
)

# Both performances should be equivalent
@test all(isapprox.(perf.train_losses, benchmark_perf.train_losses, rtol=0.001))
@test all(isapprox.(perf.test_losses, benchmark_perf.test_losses, rtol=0.001))
end

@testitem "ImitationLoss vs SPO+ (α = 1)" default_imports = false begin
#! format: noindent
include("InferOptTestUtils/InferOptTestUtils.jl")
using InferOpt, .InferOptTestUtils, LinearAlgebra, Random, Test
Random.seed!(63)

true_encoder = encoder_factory()

function spo_predictor(θ, t_true; kwargs...)
    (; θ_true) = t_true
    θ_α = θ - θ_true
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
    loss=ImitationLoss(spo_predictor; base_loss=spo_base_loss),
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
#! format: noindent
include("InferOptTestUtils/InferOptTestUtils.jl")
using InferOpt, .InferOptTestUtils, LinearAlgebra, Random, Test
Random.seed!(63)

true_encoder = encoder_factory()

function spo_predictor(θ, t_true; kwargs...)
    (; θ_true) = t_true
    θ_α = 2 .* θ - θ_true
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
    loss=ImitationLoss(spo_predictor; base_loss=spo_base_loss, α=2.0),
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
