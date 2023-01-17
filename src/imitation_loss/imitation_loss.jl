"""
    ImitationLoss{L,R,P}

Generic imitation loss: max_y ℓ(y, y̅) + θᵀ(y - y̅) - (Ω(y) - Ω(y̅)).

When `ℓ = 0`, this loss is equivalent to a [`FenchelYoungLoss`](@ref).
When `Ω = 0`, this loss is equivalent to the [`StructuredSVMLoss`](@ref).

# Fields
- `ℓ::L`: base loss, can either take (y, y_true) or (y, t_true) as input
- `Ω::R`: regularization, takes y as input
- `predictor::P`: function that computes: argmax_y ℓ(y, y̅) + θᵀ(y - y̅) - (Ω(y) - Ω(y̅)), takes (θ, y_true, kwargs...) or (θ, t_true, kwargs...) as input
"""
struct ImitationLoss{L,R,P}
    ℓ::L
    Ω::R
    predictor::P
end

function Base.show(io::IO, l::ImitationLoss)
    (; ℓ, Ω, predictor) = l
    return print(io, "ImitationLoss($ℓ, $Ω, $predictor)")
end

function prediction_and_loss(
    l::ImitationLoss, θ::AbstractArray{<:Real}, y_true::AbstractArray{<:Real}; kwargs...
)
    (; ℓ, Ω, predictor) = l
    ŷ = predictor(θ, y_true; kwargs...)
    l = ℓ(ŷ, y_true) + dot(θ, ŷ) - dot(θ, y_true) + Ω(y_true) - Ω(ŷ)
    return ŷ, l
end

# t_true must contain an `y_true` field
function prediction_and_loss(l::ImitationLoss, θ::AbstractArray{<:Real}, t_true; kwargs...)
    (; ℓ, Ω, predictor) = l
    (; y_true) = t_true
    ŷ = predictor(θ, t_true; kwargs...)
    l = ℓ(ŷ, t_true) + dot(θ, ŷ) - dot(θ, y_true) + Ω(y_true) - Ω(ŷ)
    return ŷ, l
end

## Forward pass

# function (l::ImitationLoss)(
#     θ::AbstractArray{<:Real}, y_true::AbstractArray{<:Real}; kwargs...
# )
#     _, l = prediction_and_loss(l, θ, y_true; kwargs...)
#     return l
# end

function (l::ImitationLoss)(θ::AbstractArray{<:Real}, t_true; kwargs...)
    _, l = prediction_and_loss(l, θ, t_true; kwargs...)
    return l
end

## Backward pass

function ChainRulesCore.rrule(
    l::ImitationLoss, θ::AbstractArray{<:Real}, y_true::AbstractArray{<:Real}; kwargs...
)
    ŷ, l = prediction_and_loss(l, θ, y_true; kwargs...)
    l_pullback(dl) = NoTangent(), dl .* (ŷ .- y_true), NoTangent()
    return l, l_pullback
end

function ChainRulesCore.rrule(l::ImitationLoss, θ::AbstractArray{<:Real}, t_true; kwargs...)
    (; y_true) = t_true
    ŷ, l = prediction_and_loss(l, θ, t_true; kwargs...)
    l_pullback(dl) = NoTangent(), dl .* (ŷ .- y_true), NoTangent()
    return l, l_pullback
end
