using Flux
using InferOpt
using InferOpt.Testing

function list_standard_pipelines(true_maximizer; nb_features, cost=nothing)
    pipelines = Dict{String,Vector}()

    pipelines["θ"] = [(
        encoder=Chain(Dense(nb_features, 1), dropfirstdim),
        maximizer=identity,
        loss=SPOPlusLoss(true_maximizer),
    )]

    pipelines["(θ,y)"] = [(
        encoder=Chain(Dense(nb_features, 1), dropfirstdim),
        maximizer=identity,
        loss=SPOPlusLoss(true_maximizer),
    )]

    pipelines["y"] = [
        # Perturbations
        (
            encoder=Chain(Dense(nb_features, 1), dropfirstdim),
            maximizer=identity,
            loss=FenchelYoungLoss(Perturbed(true_maximizer; ε=1.0, M=5)),
        ),
        (
            encoder=Chain(Dense(nb_features, 1), dropfirstdim),
            maximizer=Perturbed(true_maximizer; ε=1.0, M=5),
            loss=Flux.Losses.mse,
        ),
    ]

    if !isnothing(cost)
        pipelines["none"] = [(
            encoder=Chain(Dense(nb_features, 1), dropfirstdim),
            maximizer=identity,
            loss=PerturbedCost(true_maximizer, cost; ε=1.0, M=5),
        )]
    end

    return pipelines
end
