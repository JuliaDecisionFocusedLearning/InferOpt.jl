"""
    StructuredSVMLoss{L, R<:Real}

`L` should satisfy the [`IsStructuredLossFunction`](@ref) trait.

```
SSVMloss(θ, y_true) = max_y (l(y, y_true) + α [⟨θ, y⟩ - ⟨θ, y_true⟩])
```
"""
struct StructuredSVMLoss{L,R<:Real}
    predictor::L
    α::R
end

StructuredSVMLoss(predictor; α=1.0) = StructuredSVMLoss(predictor, α)

@traitfn function prediction_and_loss(
    ssvml::StructuredSVMLoss{L}, θ::AbstractArray, y_true::AbstractArray
) where {L; IsStructuredLossFunction{L}}
    y_hat = compute_maximizer(ssvml.predictor, θ, ssvml.α, y_true)
    loss = ssvml.predictor(y_hat, y_true) + dot(ssvml.α * θ, y_hat - y_true)
    return y_hat, loss
end

## Forward pass

@traitfn function (ssvml::StructuredSVMLoss{L})(
    θ::AbstractArray, y_true::AbstractArray
) where {L; IsStructuredLossFunction{L}}
    _, loss = prediction_and_loss(ssvml, θ, y_true)
    return loss
end

## Backward pass

@traitfn function ChainRulesCore.rrule(
    ssvml::StructuredSVMLoss{L}, θ::AbstractArray, y_true::AbstractArray
) where {L; IsStructuredLossFunction{L}}
    y_hat, l = prediction_and_loss(ssvml, θ, y_true)
    g = ssvml.α * (y_hat - y_true)
    ssvml_pullback(dl) = NoTangent(), dl * g, NoTangent()
    return l, ssvml_pullback
end
