"""
    SPOPlusLoss{F}

Convex surrogate of the SPO loss.

# Fields
- `maximizer::F`: linear maximizer function of the form `θ ⟼ ŷ(θ) = argmax ⟨θ,y⟩`
- `α::Float64`: convexification parameter
"""
struct SPOPlusLoss{F}
    maximizer::F
    α::Float64
end

SPOPlusLoss(maximizer; α=2.0) = SPOPlusLoss(maximizer, α)

## Forward pass

function (spol::SPOPlusLoss)(θ::AbstractArray, θ_true::AbstractArray, y_true::AbstractArray)
    @unpack maximizer, α = spol
    y_α = maximizer(α * θ - θ_true)
    l = dot(α * θ - θ_true, y_α) + dot(θ_true - α * θ, y_true)
    return l
end

function (spol::SPOPlusLoss)(θ::AbstractArray, θ_true::AbstractArray)
    y_true = spol.maximizer(θ_true)
    return spol(θ, θ_true, y_true)
end

## Backward pass

function compute_loss_and_gradient(
    spol::SPOPlusLoss, θ::AbstractArray, θ_true::AbstractArray, y_true::AbstractArray
)
    @unpack maximizer, α = spol
    y_α = maximizer(α * θ - θ_true)
    l = dot(α * θ - θ_true, y_α) + dot(θ_true - α * θ, y_true)
    return l, α * (y_α - y_true)
end

function ChainRulesCore.rrule(
    spol::SPOPlusLoss, θ::AbstractArray, θ_true::AbstractArray, y_true::AbstractArray
)
    l, g = compute_loss_and_gradient(spol, θ, θ_true, y_true)
    spol_pullback(dl) = NoTangent(), dl * g, NoTangent(), NoTangent()
    return l, spol_pullback
end

function ChainRulesCore.rrule(spol::SPOPlusLoss, θ::AbstractVector, θ_true::AbstractVector)
    y_true = spol.maximizer(θ_true)
    l, g = compute_loss_and_gradient(spol, θ, θ_true, y_true)
    spol_pullback(dl) = NoTangent(), dl * g, NoTangent()
    return l, spol_pullback
end
