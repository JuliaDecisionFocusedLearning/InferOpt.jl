using Flux
using InferOpt
using LinearAlgebra
using Random
using Test

Random.seed!(63)

## Main functions

nb_features = 5
function encoder_factory(seed=67)
    Random.seed!(seed)
    return Chain(Dense(nb_features, 1), dropfirstdim, make_positive)
end
true_encoder = encoder_factory(63)
true_maximizer(θ; kwargs...) = one_hot_argmax(θ; kwargs...)
cost(y; instance) = dot(y, -true_encoder(instance))
error_function(ŷ, y) = hamming_distance(ŷ, y)

## Pipelines

pipelines = [
    # SSVM
    (
        (
            encoder=encoder_factory(),
            maximizer=identity,
            loss=StructuredSVMLoss(ZeroOneBaseLoss()),
        ),
        (  # Equivalent to StructuredSVMLoss(ZeroOneBaseLoss())
            encoder=encoder_factory(),
            maximizer=identity,
            loss=ImitationLoss(
                ZeroOneBaseLoss(),
                y -> 0.0,
                (θ, y_true) ->
                    InferOpt.compute_maximizer(ZeroOneBaseLoss(), θ, 1.0, y_true),
            ),
        ),
    ),
    # FYL sparsemax
    (
        (
            encoder=encoder_factory(),
            maximizer=identity,
            loss=FenchelYoungLoss(sparse_argmax),
        ),
        (  # Equivalent to FenchelYoungLoss(sparse_argmax)
            encoder=encoder_factory(),
            maximizer=identity,
            loss=ImitationLoss(
                (y1, y2) -> 0.0, half_square_norm, (θ, y_true) -> sparse_argmax(θ)
            ),
        ),
    ),
    # FYL softmax
    (
        (encoder=encoder_factory(), maximizer=identity, loss=FenchelYoungLoss(soft_argmax)),
        (  # Equivalent to FenchelYoungLoss(soft_argmax)
            encoder=encoder_factory(),
            maximizer=identity,
            loss=ImitationLoss(
                (y1, y2) -> 0.0, negative_shannon_entropy, (θ, y_true) -> soft_argmax(θ)
            ),
        ),
    ),
]

spo_pipeline = (
    encoder=encoder_factory(), maximizer=identity, loss=SPOPlusLoss(true_maximizer; α=1.0)
)

function spo_base_loss(y, t_true)
    (; θ_true, y_true) = t_true
    return dot(θ_true, y_true) - dot(θ_true, y)
end

function spo_predictor(θ, t_true; kwargs...)
    (; θ_true) = t_true
    θ_α = 1 .* (θ - θ_true)
    y_α = true_maximizer(θ_α; kwargs...)
    return y_α
end

imitation_spo_pipeline = (
    encoder=encoder_factory(),
    maximizer=identity,
    loss=ImitationLoss(spo_base_loss, y -> 0.0, spo_predictor),
)

## Dataset generation

data_train, data_test = generate_dataset(
    true_encoder,
    true_maximizer;
    nb_features=nb_features,
    instance_dim=5,
    nb_instances=100,
    noise_std=0.01,
);

## Test loop

function get_perf(pipelinei)
    pipelinei = deepcopy(pipelinei)
    (; encoder, maximizer, loss) = pipelinei
    pipeline_loss_imitation_y(x, θ, y) = loss(maximizer(encoder(x)), y)
    return test_pipeline!(
        pipelinei,
        pipeline_loss_imitation_y;
        true_encoder=true_encoder,
        true_maximizer=true_maximizer,
        data_train=data_train,
        data_test=data_test,
        error_function=error_function,
        cost=cost,
        epochs=200,
        verbose=false,
        setting_name="argmax - imitation_y",
    )
end

for (pipeline1, pipeline2) in pipelines
    storage1 = get_perf(pipeline1)
    storage2 = get_perf(pipeline2)
    train_losses1 = storage1.train_losses
    test_losses1 = storage1.test_losses
    train_losses2 = storage2.train_losses
    test_losses2 = storage2.test_losses
    @testset "Imitation loss - $(pipeline1.loss)" begin
        @test all(train_losses1 .≈ train_losses2)
        @test all(test_losses1 .≈ test_losses2)
    end
end

begin
    (; encoder, maximizer, loss) = spo_pipeline
    pipeline_loss_imitation_θ(x, θ, y) = loss(maximizer(encoder(x)), θ)
    spo_storage = test_pipeline!(
        spo_pipeline,
        pipeline_loss_imitation_θ;
        true_encoder=true_encoder,
        true_maximizer=true_maximizer,
        data_train=data_train,
        data_test=data_test,
        error_function=error_function,
        cost=cost,
        epochs=100,
        verbose=false,
        setting_name="argmax - imitation_θ",
    )

    (; encoder, maximizer, loss) = imitation_spo_pipeline
    pipeline_loss_imitation_θ(x, θ, y) = loss(maximizer(encoder(x)), (; θ_true=θ, y_true=y))
    imitation_storage = test_pipeline!(
        imitation_spo_pipeline,
        pipeline_loss_imitation_θ;
        true_encoder=true_encoder,
        true_maximizer=true_maximizer,
        data_train=data_train,
        data_test=data_test,
        error_function=error_function,
        cost=cost,
        epochs=100,
        verbose=false,
        setting_name="argmax - imitation_θ - precomputed y_true",
    )

    spo_train_losses = spo_storage.train_losses
    spo_test_losses = spo_storage.test_losses
    imitation_spo_train_losses = imitation_storage.train_losses
    imitation_spo_test_losses = imitation_storage.test_losses
    @testset "Imitation loss - SPO - $(imitation_spo_pipeline.loss)" begin
        @test all(spo_train_losses .≈ imitation_spo_train_losses)
        @test all(spo_test_losses .≈ imitation_spo_test_losses)
    end
end
