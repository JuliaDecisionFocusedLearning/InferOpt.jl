"""
    ImitationLoss{L,R,P}

Generic imitation loss: max_y ℓ(y, y̅) + θᵀ(y - y̅) - (Ω(y) - Ω(y̅)).

When `ℓ = 0`, this loss is equivalent to a [`FenchelYoungLoss`](@ref).
When `Ω = 0`, this loss is equivalent to the [`StructuredSVMLoss`](@ref).

# Fields
- `base_loss::L`: base loss `ℓ`
- `Ω::R`: regularization `Ω`
- `predictor::P`: function that computes: argmax_y ℓ(y, y̅) + θᵀ(y - y̅) - (Ω(y) - Ω(y̅))
"""
struct ImitationLoss{L,R,P}
    base_loss::L
    Ω::R
    predictor::P
end

function Base.show(io::IO, l::ImitationLoss)
    (; base_loss, Ω, predictor) = l
    return print(io, "ImitationLoss($base_loss, $Ω, $predictor)")
end

function prediction_and_loss(
    l::ImitationLoss, θ::AbstractArray{<:Real}, y_true::AbstractArray{<:Real}; kwargs...
)
    (; base_loss, Ω, predictor) = l
    ŷ = predictor(θ, y_true; kwargs...)
    l = base_loss(ŷ, y_true) + dot(θ, ŷ) - dot(θ, y_true) + Ω(y_true) - Ω(ŷ)
    return ŷ, l
end

## Forward pass

function (l::ImitationLoss)(
    θ::AbstractArray{<:Real}, y_true::AbstractArray{<:Real}; kwargs...
)
    _, l = prediction_and_loss(l, θ, y_true; kwargs...)
    return l
end

## Backward pass

function ChainRulesCore.rrule(
    l::ImitationLoss, θ::AbstractArray{<:Real}, y_true::AbstractArray{<:Real}; kwargs...
)
    ŷ, l = prediction_and_loss(l, θ, y_true; kwargs...)
    g = ŷ .- y_true
    l_pullback(dl) = NoTangent(), dl * g, NoTangent()
    return l, l_pullback
end
