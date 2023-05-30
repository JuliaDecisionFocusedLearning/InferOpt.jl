@testitem "Argmax - imit - SPO+ (θ)" default_imports = false begin
#! format: noindent
include("InferOptTestUtils/InferOptTestUtils.jl")
using InferOpt, .InferOptTestUtils, Random
Random.seed!(63)

test_pipeline!(
    PipelineLossImitationθ;
    instance_dim=5,
    true_maximizer=one_hot_argmax,
    maximizer=identity,
    loss=SPOPlusLoss(one_hot_argmax),
    error_function=hamming_distance,
)
end

@testitem "Argmax - imit - SPO+ (θ & y)" default_imports = false begin
#! format: noindent
include("InferOptTestUtils/InferOptTestUtils.jl")
using InferOpt, .InferOptTestUtils, Random
Random.seed!(63)

test_pipeline!(
    PipelineLossImitationθy;
    instance_dim=5,
    true_maximizer=one_hot_argmax,
    maximizer=identity,
    loss=SPOPlusLoss(one_hot_argmax),
    error_function=hamming_distance,
)
end

@testitem "Argmax - imit - SSVM" default_imports = false begin
#! format: noindent
include("InferOptTestUtils/InferOptTestUtils.jl")
using InferOpt, .InferOptTestUtils, Random
Random.seed!(63)

test_pipeline!(
    PipelineLossImitation;
    instance_dim=5,
    true_maximizer=one_hot_argmax,
    maximizer=identity,
    loss=StructuredSVMLoss(ZeroOneBaseLoss()),
    error_function=hamming_distance,
)
end

@testitem "Argmax - imit - MSE sparse argmax" default_imports = false begin
#! format: noindent
include("InferOptTestUtils/InferOptTestUtils.jl")
using InferOpt, .InferOptTestUtils, Random
Random.seed!(63)

test_pipeline!(
    PipelineLossImitation;
    instance_dim=5,
    true_maximizer=one_hot_argmax,
    maximizer=sparse_argmax,
    loss=mse,
    error_function=hamming_distance,
)
end

@testitem "Argmax - imit - MSE soft argmax" default_imports = false begin
#! format: noindent
include("InferOptTestUtils/InferOptTestUtils.jl")
using InferOpt, .InferOptTestUtils, Random
Random.seed!(63)

test_pipeline!(
    PipelineLossImitation;
    instance_dim=5,
    true_maximizer=one_hot_argmax,
    maximizer=soft_argmax,
    loss=mse,
    error_function=hamming_distance,
)
end

@testitem "Argmax - imit - MSE PerturbedAdditive" default_imports = false begin
#! format: noindent
include("InferOptTestUtils/InferOptTestUtils.jl")
using InferOpt, .InferOptTestUtils, Random
Random.seed!(63)

test_pipeline!(
    PipelineLossImitation;
    instance_dim=5,
    true_maximizer=one_hot_argmax,
    maximizer=PerturbedAdditive(one_hot_argmax; ε=1.0, nb_samples=10),
    loss=mse,
    error_function=hamming_distance,
)
end

@testitem "Argmax - imit - MSE PerturbedMultiplicative" default_imports = false begin
#! format: noindent
include("InferOptTestUtils/InferOptTestUtils.jl")
using InferOpt, .InferOptTestUtils, Random
Random.seed!(63)

test_pipeline!(
    PipelineLossImitation;
    instance_dim=5,
    true_maximizer=one_hot_argmax,
    maximizer=PerturbedMultiplicative(one_hot_argmax; ε=1.0, nb_samples=10),
    loss=mse,
    error_function=hamming_distance,
)
end

@testitem "Argmax - imit - MSE RegularizedGeneric" default_imports = false begin
#! format: noindent
include("InferOptTestUtils/InferOptTestUtils.jl")
using InferOpt, .InferOptTestUtils, Random
Random.seed!(63)

test_pipeline!(
    PipelineLossImitation;
    instance_dim=5,
    true_maximizer=one_hot_argmax,
    maximizer=RegularizedGeneric(one_hot_argmax, half_square_norm, identity),
    loss=mse,
    error_function=hamming_distance,
)
end

@testitem "Argmax - imit - FYL sparse argmax" default_imports = false begin
#! format: noindent
include("InferOptTestUtils/InferOptTestUtils.jl")
using InferOpt, .InferOptTestUtils, Random
Random.seed!(63)

test_pipeline!(
    PipelineLossImitation;
    instance_dim=5,
    true_maximizer=one_hot_argmax,
    maximizer=identity,
    loss=FenchelYoungLoss(sparse_argmax),
    error_function=hamming_distance,
)
end

@testitem "Argmax - imit - FYL soft argmax" default_imports = false begin
#! format: noindent
include("InferOptTestUtils/InferOptTestUtils.jl")
using InferOpt, .InferOptTestUtils, Random
Random.seed!(63)

test_pipeline!(
    PipelineLossImitation;
    instance_dim=5,
    true_maximizer=one_hot_argmax,
    maximizer=identity,
    loss=FenchelYoungLoss(soft_argmax),
    error_function=hamming_distance,
)
end

@testitem "Argmax - imit - FYL PerturbedAdditive" default_imports = false begin
#! format: noindent
include("InferOptTestUtils/InferOptTestUtils.jl")
using InferOpt, .InferOptTestUtils, Random
Random.seed!(63)

test_pipeline!(
    PipelineLossImitation;
    instance_dim=5,
    true_maximizer=one_hot_argmax,
    maximizer=identity,
    loss=FenchelYoungLoss(PerturbedAdditive(one_hot_argmax; ε=1.0, nb_samples=5)),
    error_function=hamming_distance,
)
end

@testitem "Argmax - imit - FYL PerturbedMultiplicative" default_imports = false begin
#! format: noindent
include("InferOptTestUtils/InferOptTestUtils.jl")
using InferOpt, .InferOptTestUtils, Random
Random.seed!(63)

test_pipeline!(
    PipelineLossImitation;
    instance_dim=5,
    true_maximizer=one_hot_argmax,
    maximizer=identity,
    loss=FenchelYoungLoss(PerturbedMultiplicative(one_hot_argmax; ε=1.0, nb_samples=5)),
    error_function=hamming_distance,
)
end

@testitem "Argmax - imit - FYL RegularizedGeneric" default_imports = false begin
#! format: noindent
include("InferOptTestUtils/InferOptTestUtils.jl")
using InferOpt, .InferOptTestUtils, Random
Random.seed!(63)

test_pipeline!(
    PipelineLossImitation;
    instance_dim=5,
    true_maximizer=one_hot_argmax,
    maximizer=identity,
    loss=FenchelYoungLoss(
        RegularizedGeneric(one_hot_argmax, half_square_norm, identity)
    ),
    error_function=hamming_distance,
)
end

@testitem "Argmax - exp - Pushforward PerturbedAdditive" default_imports = false begin
#! format: noindent
include("InferOptTestUtils/InferOptTestUtils.jl")
using InferOpt, .InferOptTestUtils, LinearAlgebra, Random
Random.seed!(63)

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

@testitem "Argmax - exp - Pushforward PerturbedMultiplicative" default_imports = false begin
#! format: noindent
include("InferOptTestUtils/InferOptTestUtils.jl")
using InferOpt, .InferOptTestUtils, LinearAlgebra, Random
Random.seed!(63)

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

@testitem "Argmax - exp - Pushforward RegularizedGeneric" default_imports = false begin
#! format: noindent
include("InferOptTestUtils/InferOptTestUtils.jl")
using InferOpt, .InferOptTestUtils, LinearAlgebra, Random
Random.seed!(63)

true_encoder = encoder_factory()
cost(y; instance) = dot(y, -true_encoder(instance))
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
)
end
