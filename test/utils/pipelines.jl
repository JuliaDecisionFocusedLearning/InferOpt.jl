using Flux
using InferOpt
using InferOpt.Testing

function list_standard_pipelines(true_maximizer; nb_features, cost=nothing)
    pipelines = Dict{String,Vector}()
    # SPO+
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
        # Perturbations with FYL
        (
            encoder=Chain(Dense(nb_features, 1), dropfirstdim),
            maximizer=identity,
            loss=FenchelYoungLoss(PerturbedNormal(true_maximizer; ε=1.0, M=5)),
        ),
        (
            encoder=Chain(Dense(nb_features, 1), dropfirstdim),
            maximizer=identity,
            loss=FenchelYoungLoss(PerturbedLogNormal(true_maximizer; ε=0.1, M=5)),
        ),
        # Perturbation with MSE
        (
            encoder=Chain(Dense(nb_features, 1), dropfirstdim),
            maximizer=PerturbedNormal(true_maximizer; ε=1.0, M=5),
            loss=Flux.Losses.mse,
        ),
    ]

    if !isnothing(cost)
        pipelines["none"] = [(
            encoder=Chain(Dense(nb_features, 1), dropfirstdim),
            maximizer=identity,
            loss=PerturbedCost(PerturbedNormal(true_maximizer; ε=1.0, M=5), cost),
        )]
    end

    return pipelines
end
