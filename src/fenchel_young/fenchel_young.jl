"""
    FenchelYoungLoss{P}

Fenchel-Young loss associated with a given regularized prediction function.

# Fields
- `predictor::P`: prediction function, usually of the form `θ ⟼ ŷ(θ) = argmax ⟨θ,y⟩ - Ω(y)`
"""
struct FenchelYoungLoss{P}
    predictor::P
end

function Base.show(io::IO, fyl::FenchelYoungLoss)
    (; predictor) = fyl
    print(io, "FenchelYoungLoss($predictor)")
end

## Forward pass

@traitfn function prediction_and_loss(
    fyl::FenchelYoungLoss{P}, θ::AbstractArray{<:Real}, y::AbstractArray{<:Real}; kwargs...
) where {P; IsRegularizedPrediction{P}}
    (; predictor) = fyl
    ŷ = predictor(θ; kwargs...)
    fy = compute_regularization(predictor, y) - dot(θ, y)
    fŷ = compute_regularization(predictor, ŷ) - dot(θ, ŷ)
    l = fy - fŷ
    return ŷ, l
end

function prediction_and_loss(
    fyl::FenchelYoungLoss{P}, θ::AbstractArray{<:Real}, y::AbstractArray{<:Real}; kwargs...
) where {P<:AbstractPerturbed}
    (; predictor) = fyl
    ŷ, Fθ = compute_y_and_F(predictor, θ; kwargs...)
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
