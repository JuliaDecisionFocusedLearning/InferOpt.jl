"""
    StructuredSVMLoss{L}

Loss associated with the Structured Support Vector Machine.

`ℓ(θ, y_true) = max_y {δ(y, y_true) + α θᵀ(y - y_true)}`

# Fields
- `base_loss::L`:  of the `IsBaseLoss` trait
- `α::Float64`

Reference: <http://www.nowozin.net/sebastian/papers/nowozin2011structured-tutorial.pdf> (Chapter 6)
"""
struct StructuredSVMLoss{L}
    base_loss::L
    α::Float64
end

function Base.show(io::IO, ssvml::StructuredSVMLoss)
    (; base_loss, α) = ssvml
    return print(io, "StructuredSVMLoss($base_loss, $α)")
end

StructuredSVMLoss(base_loss; α=1.0) = StructuredSVMLoss(base_loss, float(α))

@traitfn function prediction_and_loss(
    ssvml::StructuredSVMLoss{L},
    θ::AbstractArray{<:Real},
    y_true::AbstractArray{<:Real};
    kwargs...,
) where {L; IsBaseLoss{L}}
    (; base_loss, α) = ssvml
    ŷ = compute_maximizer(base_loss, θ, α, y_true; kwargs...)
    l = base_loss(ŷ, y_true) + α * (dot(θ, ŷ) - dot(θ, y_true))
    return ŷ, l
end

## Forward pass

@traitfn function (ssvml::StructuredSVMLoss{L})(
    θ::AbstractArray{<:Real}, y_true::AbstractArray{<:Real}; kwargs...
) where {L; IsBaseLoss{L}}
    _, l = prediction_and_loss(ssvml, θ, y_true; kwargs...)
    return l
end

## Backward pass

@traitfn function ChainRulesCore.rrule(
    ssvml::StructuredSVMLoss{L},
    θ::AbstractArray{<:Real},
    y_true::AbstractArray{<:Real};
    kwargs...,
) where {L; IsBaseLoss{L}}
    (; α) = ssvml
    ŷ, l = prediction_and_loss(ssvml, θ, y_true; kwargs...)
    g = α .* (ŷ .- y_true)
    ssvml_pullback(dl) = NoTangent(), dl * g, NoTangent()
    return l, ssvml_pullback
end
