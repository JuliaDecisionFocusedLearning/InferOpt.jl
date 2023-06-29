"""
    FenchelYoungLoss{P}

Fenchel-Young loss associated with a given regularized prediction function.

# Fields
- `predictor::P`: prediction function of the form `ŷ(θ) = argmax {θᵀy - Ω(y)}`

Reference: <https://arxiv.org/abs/1901.02324>
"""
struct FenchelYoungLoss{P}
    predictor::P
end

function Base.show(io::IO, fyl::FenchelYoungLoss)
    (; predictor) = fyl
    return print(io, "FenchelYoungLoss($predictor)")
end

## Forward pass

function (fyl::FenchelYoungLoss)(θ::AbstractArray, y_true::AbstractArray; kwargs...)
    l, _ = fenchel_young_loss_and_grad(fyl, θ, y_true; kwargs...)
    return l
end

@traitfn function fenchel_young_loss_and_grad(
    fyl::FenchelYoungLoss{P}, θ::AbstractArray, y_true::AbstractArray; kwargs...
) where {P; IsRegularized{P}}
    (; predictor) = fyl
    ŷ = predictor(θ; kwargs...)
    Ωy_true = compute_regularization(predictor, y_true)
    Ωŷ = compute_regularization(predictor, ŷ)
    l = (Ωy_true - dot(θ, y_true)) - (Ωŷ - dot(θ, ŷ))
    g = ŷ - y_true
    return l, g
end

function fenchel_young_loss_and_grad(
    fyl::FenchelYoungLoss{P}, θ::AbstractArray, y_true::AbstractArray; kwargs...
) where {P<:AbstractPerturbed}
    (; predictor) = fyl
    F, almost_ŷ = fenchel_young_F_and_first_part_of_grad(predictor, θ; kwargs...)
    l = F - dot(θ, y_true)
    g = almost_ŷ - y_true
    return l, g
end

function fenchel_young_loss_and_grad(
    fyl::FenchelYoungLoss{P},
    θ::AbstractArray{<:Real},
    y_true::AbstractArray{<:Real};
    kwargs...,
) where {P<:AbstractPerturbed{<:GeneralizedMaximizer}}
    (; predictor) = fyl
    F, almost_g_of_ŷ = fenchel_young_F_and_first_part_of_grad(predictor, θ; kwargs...)
    l = F - objective_value(predictor.maximizer, θ, y_true; kwargs...)
    # @info size(almost_g_of_ŷ), size(y_true), size(predictor.maximizer.g(y_true))
    g = almost_g_of_ŷ - predictor.maximizer.g(y_true)
    return l, g
end

## Backward pass

function ChainRulesCore.rrule(
    fyl::FenchelYoungLoss, θ::AbstractArray, y_true::AbstractArray; kwargs...
)
    l, g = fenchel_young_loss_and_grad(fyl, θ, y_true; kwargs...)
    fyl_pullback(dl) = NoTangent(), dl * g, NoTangent()
    return l, fyl_pullback
end
