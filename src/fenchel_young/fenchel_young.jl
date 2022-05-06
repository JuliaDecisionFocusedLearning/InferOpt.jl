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
    fyl::FenchelYoungLoss{P}, θ::AbstractArray, y::AbstractArray; kwargs...
) where {P; IsRegularizedPrediction{P}}
    ŷ = fyl.predictor(θ; kwargs...)
    Ω(z) = compute_regularization(fyl.predictor, z)
    l = dot(θ, ŷ - y) - (Ω(ŷ) - Ω(y))
    return ŷ, l
end

function prediction_and_loss(
    fyl::FenchelYoungLoss{P}, θ::AbstractArray, y::AbstractArray; kwargs...
) where {P<:AbstractPerturbed}
    ŷ, Fθ = compute_y_and_Fθ(fyl.predictor, θ; kwargs...)
    l = Fθ - dot(θ, y)
    return ŷ, l
end

## Forward pass

function (fyl::FenchelYoungLoss)(θ::AbstractArray, y::AbstractArray; kwargs...)
    _, l = prediction_and_loss(fyl, θ, y; kwargs...)
    return l
end

## Backward pass

function ChainRulesCore.rrule(
    fyl::FenchelYoungLoss, θ::AbstractArray, y::AbstractArray; kwargs...
)
    ŷ, l = prediction_and_loss(fyl, θ, y; kwargs...)
    fyl_pullback(dl) = NoTangent(), dl * (ŷ - y), NoTangent()
    return l, fyl_pullback
end
