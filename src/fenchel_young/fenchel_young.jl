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

function (fyl::FenchelYoungLoss)(
    θ::AbstractArray{<:Real}, y::AbstractArray{<:Real}; kwargs...
)
    _, l = prediction_and_loss(fyl, θ, y; kwargs...)
    return l
end

@traitfn function prediction_and_loss(
    fyl::FenchelYoungLoss{P},
    θ::AbstractArray{<:Real},
    y_true::AbstractArray{<:Real};
    kwargs...,
) where {P; IsRegularized{P}}
    (; predictor) = fyl
    ŷ = predictor(θ; kwargs...)
    Ωy_true = compute_regularization(predictor, y_true)
    Ωŷ = compute_regularization(predictor, ŷ)
    l = (Ωy_true - dot(θ, y_true)) - (Ωŷ - dot(θ, ŷ))
    return ŷ, l
end

function prediction_and_loss(
    fyl::FenchelYoungLoss{P}, θ::AbstractArray{<:Real}, y::AbstractArray{<:Real}; kwargs...
) where {P<:AbstractPerturbed}
    (; predictor) = fyl
    ŷ, F = compute_y_and_F(predictor, θ; kwargs...)
    l = F - dot(θ, y)
    return ŷ, l
end

## Backward pass

function ChainRulesCore.rrule(
    fyl::FenchelYoungLoss, θ::AbstractArray{<:Real}, y::AbstractArray{<:Real}; kwargs...
)
    ŷ, l = prediction_and_loss(fyl, θ, y; kwargs...)
    fyl_pullback(dl) = NoTangent(), dl * (ŷ - y), NoTangent()
    return l, fyl_pullback
end
