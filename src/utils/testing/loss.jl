function define_flux_loss(encoder, maximizer, loss, target)
    flux_loss_none(x, θ, y) = loss(maximizer(encoder(x)); instance=x)
    flux_loss_θ(x, θ, y) = loss(maximizer(encoder(x)), θ)
    flux_loss_y(x, θ, y) = loss(maximizer(encoder(x)), y)
    flux_loss_θy(x, θ, y) = loss(maximizer(encoder(x)), θ, y)

    flux_losses = Dict(
        "none" => flux_loss_none,
        "θ" => flux_loss_θ,
        "y" => flux_loss_y,
        "(θ,y)" => flux_loss_θy,
    )

    return flux_losses[target]
end
