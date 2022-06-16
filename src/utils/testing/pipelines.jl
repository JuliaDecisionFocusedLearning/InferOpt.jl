function list_standard_pipelines(encoder_factory, true_maximizer; cost=nothing)
    pipelines = Dict{String,Vector}()
    # SPO+
    pipelines["θ"] = [(
        encoder=encoder_factory(),
        maximizer=identity,
        loss=SPOPlusLoss(true_maximizer),
    )]

    pipelines["(θ,y)"] = [(
        encoder=encoder_factory(),
        maximizer=identity,
        loss=SPOPlusLoss(true_maximizer),
    )]
    # Perturbations
    pipelines["y"] = [
        # Fenchel-Young loss (test forward pass)
        (
            encoder=encoder_factory(),
            maximizer=identity,
            loss=FenchelYoungLoss(PerturbedNormal(true_maximizer; ε=1.0, M=5)),
        ),
        (
            encoder=encoder_factory(),
            maximizer=identity,
            loss=FenchelYoungLoss(PerturbedLogNormal(true_maximizer; ε=0.1, M=5)),
        ),
        # Other differentiable loss (test backward pass)
        (
            encoder=encoder_factory(),
            maximizer=PerturbedNormal(true_maximizer; ε=1.0, M=5),
            loss=(ŷ, y) -> half_square_norm(y - ŷ),
        ),
    ]
    # Learning by experience
    if !isnothing(cost)
        pipelines["none"] = [
            (
            encoder=encoder_factory(),
            maximizer=identity,
            loss=PerturbedCost(PerturbedNormal(true_maximizer; ε=1.0, M=5), cost),
        ),
        (
            encoder=encoder_factory(),
            maximizer=identity,
            loss=PerturbedCost(PerturbedLogNormal(true_maximizer; ε=0.7, M=5), cost),
        ),
        ]
    end

    return pipelines
end

function define_pipeline_loss(encoder, maximizer, loss, target)
    pipeline_loss_none(x, θ, y) = loss(maximizer(encoder(x)); instance=x)
    pipeline_loss_θ(x, θ, y) = loss(maximizer(encoder(x)), θ)
    pipeline_loss_y(x, θ, y) = loss(maximizer(encoder(x)), y)
    pipeline_loss_θy(x, θ, y) = loss(maximizer(encoder(x)), θ, y)

    pipeline_losses = Dict(
        "none" => pipeline_loss_none,
        "θ" => pipeline_loss_θ,
        "y" => pipeline_loss_y,
        "(θ,y)" => pipeline_loss_θy,
    )

    return pipeline_losses[target]
end
