using Distributions
using Flux
using InferOpt
using LinearAlgebra
using ProgressMeter
using Random
using Test
using UnicodePlots

Random.seed!(63)

## Dimensions and parameters

nb_features = 5
nb_instances = 100
instance_dim = 20
noise_std = 0.02

epochs = 200
show_plots = false

## Main functions

true_model = Chain(Dense(nb_features, 1), InferOpt.dropfirstdim)
optimizer = ranking
error_function = Flux.Losses.mse
cost(y; instance) = dot(y, -true_model(instance))

## Dataset generation

data_train, data_test = InferOpt.generate_dataset(
    true_model,
    optimizer;
    nb_features=nb_features,
    instance_dim=instance_dim,
    nb_instances=nb_instances,
    noise_std=noise_std,
);

## Pipelines

pipelines = Dict(
    "none" => [(
        model=Chain(Dense(nb_features, 1), InferOpt.dropfirstdim),
        loss=PerturbedCost(ranking, cost; ε=1.0, M=10),
    )],
    "θ" => [(
        model=Chain(Dense(nb_features, 1), InferOpt.dropfirstdim),
        loss=SPOPlusLoss(ranking),
    )],
    "(θ,y)" => [(
        model=Chain(Dense(nb_features, 1), InferOpt.dropfirstdim),
        loss=SPOPlusLoss(ranking),
    )],
    "y" => [
        (
            model=Chain(Dense(nb_features, 1), InferOpt.dropfirstdim),
            loss=FenchelYoungLoss(Perturbed(ranking; ε=1.0, M=10)),
        ),
        (
            model=Chain(Dense(nb_features, 1), InferOpt.dropfirstdim),
            loss=FenchelYoungLoss(
                PerturbedGeneric(
                    ranking; noise_dist=θ -> MultivariateNormal(θ, 0.1^2 * I), M=10
                ),
            ),
        ),
        (
            model=Chain(
                Dense(nb_features, 1),
                InferOpt.dropfirstdim,
                Perturbed(ranking; ε=1.0, M=10),
            ),
            loss=Flux.Losses.mse,
        ),
        (
            model=Chain(
                Dense(nb_features, 1),
                InferOpt.dropfirstdim,
                Interpolation(ranking; λ=10.0),
            ),
            loss=Flux.Losses.mse,
        ),
    ],
)

## Test loop

for target in ["θ", "(θ,y)", "y"], (; model, loss) in pipelines[target]
    @info "Testing argmax" target model loss
    flux_loss = InferOpt.define_flux_loss(model, loss, target)

    ## Optimization

    opt = ADAM()
    perf_storage = InferOpt.init_perf()

    @showprogress for _ in 1:epochs
        InferOpt.update_perf!(
            perf_storage;
            data_train=data_train,
            data_test=data_test,
            true_model=true_model,
            model=model,
            optimizer=optimizer,
            flux_loss=flux_loss,
            error_function=error_function,
            cost=cost,
        )
        Flux.train!(flux_loss, Flux.params(model), zip(data_train...), opt)
    end

    ## Evaluation

    if show_plots
        InferOpt.plot_perf(perf_storage)
    end
    InferOpt.test_perf(perf_storage; test_name="$target - $model - $loss")
end
