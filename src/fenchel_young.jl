"""
    FenchelYoungLoss{P}

Fenchel-Young loss associated with a given regularized prediction function.

# Fields
- `predictor::P`: prediction function, usually of the form `θ ⟼ ŷ(θ) = argmax ⟨θ,y⟩ - Ω(y)`
"""
struct FenchelYoungLoss{P}
    predictor::P
end

## Forward pass

@traitfn function prediction_and_loss(
    fyl::FenchelYoungLoss{P}, θ::AbstractVector, y::AbstractVector
) where {P; IsRegularizedPrediction{P}}
    ŷ = fyl.predictor(θ)
    Ω(z) = compute_regularization(fyl.predictor, z)
    l = θ' * (ŷ - y) - (Ω(ŷ) - Ω(y))
    return ŷ, l
end

function prediction_and_loss(
    fyl::FenchelYoungLoss{P}, θ::AbstractVector, y::AbstractVector; kwargs...
) where {P<:Perturbed}
    ŷ = fyl.predictor(θ; kwargs...)
    l = θ' * (ŷ - y)  # TODO: missing term?
    return ŷ, l
end

## Forward pass

function (fyl::FenchelYoungLoss)(θ::AbstractVector, y::AbstractVector; kwargs...)
    _, l = prediction_and_loss(fyl, θ, y; kwargs...)
    return l
end

## Backward pass

function ChainRulesCore.rrule(
    fyl::FenchelYoungLoss, θ::AbstractVector, y::AbstractVector; kwargs...
)
    ŷ, l = prediction_and_loss(fyl, θ, y; kwargs...)
    fyl_pullback(dl) = NoTangent(), dl * (ŷ - y), NoTangent()
    return l, fyl_pullback
end
