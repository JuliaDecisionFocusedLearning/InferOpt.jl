"""
    SPOPlusLoss <: AbstractLossLayer

Convex surrogate of the Smart "Predict-then-Optimize" loss.

# Fields
- `maximizer`: linear maximizer function of the form `θ -> ŷ(θ) = argmax θᵀy`
- `α::Float64`: convexification parameter, default = 2.0

Reference: <https://arxiv.org/abs/1710.08005>
"""
struct SPOPlusLoss{F} <: AbstractLossLayer
    maximizer::F
    α::Float64
end

function Base.show(io::IO, spol::SPOPlusLoss)
    (; maximizer, α) = spol
    return print(io, "SPOPlusLoss($maximizer, $α)")
end

"""
    SPOPlusLoss(maximizer; α=2.0)
"""
SPOPlusLoss(maximizer; α=2.0) = SPOPlusLoss(maximizer, float(α))

## Forward pass

"""
    (spol::SPOPlusLoss)(θ, θ_true, y_true; kwargs...)
"""
function (spol::SPOPlusLoss)(
    θ::AbstractArray, θ_true::AbstractArray, y_true::AbstractArray; kwargs...
)
    (; maximizer, α) = spol
    θ_α = α * θ - θ_true
    y_α = maximizer(θ_α; kwargs...)
    l = dot(θ_α, y_α) - dot(θ_α, y_true)
    return l
end

"""
    (spol::SPOPlusLoss)(θ, θ_true; kwargs...)
"""
function (spol::SPOPlusLoss)(θ::AbstractArray, θ_true::AbstractArray; kwargs...)
    y_true = spol.maximizer(θ_true; kwargs...)
    return spol(θ, θ_true, y_true)
end

## Backward pass

function compute_loss_and_gradient(
    spol::SPOPlusLoss,
    θ::AbstractArray,
    θ_true::AbstractArray,
    y_true::AbstractArray;
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
    θ::AbstractArray,
    θ_true::AbstractArray,
    y_true::AbstractArray;
    kwargs...,
)
    l, g = compute_loss_and_gradient(spol, θ, θ_true, y_true; kwargs...)
    spol_pullback(dl) = NoTangent(), dl * g, NoTangent(), NoTangent()
    return l, spol_pullback
end

function ChainRulesCore.rrule(
    spol::SPOPlusLoss, θ::AbstractArray, θ_true::AbstractArray; kwargs...
)
    y_true = spol.maximizer(θ_true; kwargs...)
    l, g = compute_loss_and_gradient(spol, θ, θ_true, y_true; kwargs...)
    spol_pullback(dl) = NoTangent(), dl * g, NoTangent()
    return l, spol_pullback
end
