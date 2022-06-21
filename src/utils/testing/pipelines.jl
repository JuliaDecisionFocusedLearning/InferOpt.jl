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
