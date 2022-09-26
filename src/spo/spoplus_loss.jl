"""
    SPOPlusLoss{F}

Convex surrogate of the Smart "Predict-then-Optimize" loss.

# Fields
- `maximizer::F`: linear maximizer function of the form `θ ⟼ ŷ(θ) = argmax θᵀy`
- `α::Float64`: convexification parameter

Reference: <https://arxiv.org/abs/1710.08005>
"""
struct SPOPlusLoss{F}
    maximizer::F
    α::Float64
end

function Base.show(io::IO, spol::SPOPlusLoss)
    (; maximizer, α) = spol
    return print(io, "SPOPlusLoss($maximizer, $α)")
end

SPOPlusLoss(maximizer; α=2.0) = SPOPlusLoss(maximizer, float(α))

## Forward pass

function (spol::SPOPlusLoss)(
    θ::AbstractArray{<:Real},
    θ_true::AbstractArray{<:Real},
    y_true::AbstractArray{<:Real};
    kwargs...,
)
    (; maximizer, α) = spol
    θ_α = α * θ - θ_true
    y_α = maximizer(θ_α; kwargs...)
    l = dot(θ_α, y_α) - dot(θ_α, y_true)
    return l
end

function (spol::SPOPlusLoss)(
    θ::AbstractArray{<:Real}, θ_true::AbstractArray{<:Real}; kwargs...
)
    y_true = spol.maximizer(θ_true; kwargs...)
    return spol(θ, θ_true, y_true)
end

## Backward pass

function compute_loss_and_gradient(
    spol::SPOPlusLoss,
    θ::AbstractArray{<:Real},
    θ_true::AbstractArray{<:Real},
    y_true::AbstractArray{<:Real};
    kwargs...,
)
    (; maximizer, α) = spol
    θ_α = α * θ - θ_true
    y_α = maximizer(θ_α; kwargs...)
    l = dot(θ_α, y_α) - dot(θ_α, y_true)
    return l, α .* (y_α .- y_true)
end

function ChainRulesCore.rrule(
    spol::SPOPlusLoss,
    θ::AbstractArray{<:Real},
    θ_true::AbstractArray{<:Real},
    y_true::AbstractArray{<:Real};
    kwargs...,
)
    l, g = compute_loss_and_gradient(spol, θ, θ_true, y_true; kwargs...)
    spol_pullback(dl) = NoTangent(), dl * g, NoTangent(), NoTangent()
    return l, spol_pullback
end

function ChainRulesCore.rrule(
    spol::SPOPlusLoss, θ::AbstractArray{<:Real}, θ_true::AbstractArray{<:Real}; kwargs...
)
    y_true = spol.maximizer(θ_true; kwargs...)
    l, g = compute_loss_and_gradient(spol, θ, θ_true, y_true; kwargs...)
    spol_pullback(dl) = NoTangent(), dl * g, NoTangent()
    return l, spol_pullback
end
