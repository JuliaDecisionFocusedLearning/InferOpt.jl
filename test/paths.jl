using Distributions
using Flux
using Graphs
using InferOpt
using LinearAlgebra
using ProgressMeter
using Random
using Test
using UnicodePlots

Random.seed!(63)

include("pipelines.jl")

## Dimensions and parameters

nb_features = 5
nb_instances = 50
instance_dim = (10, 20)
noise_std = 0.02

epochs = 500
show_plots = true

## Main functions

true_encoder = Chain(Dense(nb_features, 1), InferOpt.dropfirstdim)

function true_maximizer(θ::AbstractMatrix; kwargs...)
    g = AcyclicGridGraph(-θ)
    return grid_shortest_path(g, 1, nv(g))
end

cost(y; instance) = dot(y, -true_encoder(instance))
error_function(y1, y2) = Flux.Losses.mse(y1, y2)

## Dataset generation

data_train, data_test = InferOpt.generate_dataset(
    true_encoder,
    true_maximizer;
    nb_features=nb_features,
    instance_dim=instance_dim,
    nb_instances=nb_instances,
    noise_std=noise_std,
);

## Pipelines

pipelines = list_standard_pipelines(true_maximizer; cost=nothing, nb_features=nb_features)

delete!(pipelines, "θ")
delete!(pipelines, "(θ,y)")

## Test loop

for target in keys(pipelines), pipeline in pipelines[target]
    (; encoder, maximizer, loss) = pipeline
    flux_loss = InferOpt.define_flux_loss(encoder, maximizer, loss, target)
    @info "Testing paths" target encoder maximizer loss

    ## Optimization

    opt = ADAM()
    perf_storage = InferOpt.init_perf()

    @showprogress for _ in 1:epochs
        InferOpt.update_perf!(
            perf_storage;
            data_train=data_train,
            data_test=data_test,
            true_encoder=true_encoder,
            encoder=encoder,
            true_maximizer=true_maximizer,
            flux_loss=flux_loss,
            error_function=error_function,
            cost=cost,
        )
        Flux.train!(flux_loss, Flux.params(encoder), zip(data_train...), opt)
    end

    ## Evaluation

    if show_plots
        InferOpt.plot_perf(perf_storage)
    end
    InferOpt.test_perf(perf_storage; test_name="$target - $maximizer - $loss")
end
