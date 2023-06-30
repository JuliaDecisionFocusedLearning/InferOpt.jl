"""
    StructuredSVMLoss <: AbstractLossLayer

Loss associated with the Structured Support Vector Machine, defined by
```
L(θ, y_true) = max_y {δ(y, y_true) + α θᵀ(y - y_true)}
```

Reference: <http://www.nowozin.net/sebastian/papers/nowozin2011structured-tutorial.pdf> (Chapter 6)

# Fields

- `aux_loss_maximizer::M`: function of `(θ, y_true, α)` that computes the argmax in the problem above
- `δ::L`: base loss function
- `α::Float64`: hyperparameter with a default value of 1.0
"""
struct StructuredSVMLoss{M,L} <: AbstractLossLayer
    aux_loss_maximizer::M
    δ::L
    α::Float64
end

"""
    StructuredSVMLoss(; aux_loss_maximizer, δ, α=1.0)

Explicit constructor with keyword arguments.
"""
function StructuredSVMLoss(; aux_loss_maximizer, δ, α=1.0)
    return StructuredSVMLoss(aux_loss_maximizer, δ, float(α))
end

function Base.show(io::IO, ssvml::StructuredSVMLoss)
    (; aux_loss_maximizer, δ, α) = ssvml
    return print(io, "StructuredSVMLoss($aux_loss_maximizer, $δ, $α)")
end

function prediction_and_loss(
    ssvml::StructuredSVMLoss, θ::AbstractArray, y_true::AbstractArray; kwargs...
)
    (; aux_loss_maximizer, δ, α) = ssvml
    ŷ = aux_loss_maximizer(θ, y_true, α; kwargs...)
    l = δ(ŷ, y_true) + α * (dot(θ, ŷ) - dot(θ, y_true))
    return ŷ, l
end

## Forward pass

function (ssvml::StructuredSVMLoss)(θ::AbstractArray, y_true::AbstractArray; kwargs...)
    _, l = prediction_and_loss(ssvml, θ, y_true; kwargs...)
    return l
end

## Backward pass

function ChainRulesCore.rrule(
    ssvml::StructuredSVMLoss, θ::AbstractArray, y_true::AbstractArray; kwargs...
)
    (; α) = ssvml
    ŷ, l = prediction_and_loss(ssvml, θ, y_true; kwargs...)
    g = α .* (ŷ .- y_true)
    ssvml_pullback(dl) = NoTangent(), dl * g, NoTangent()
    return l, ssvml_pullback
end
