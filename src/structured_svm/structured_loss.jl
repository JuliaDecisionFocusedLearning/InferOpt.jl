"""
    IsStructuredLossFunction{L}

Trait-based interface for structured loss functions of the form `(y, y_true) ⟼ l(y, y_true).
You need a StructuredLossFunction in order build StructuredSVM losses.

For a type `L` to comply with this interface, the following methods must exist:
- `(loss::L)(y, y_true)`
- `compute_maximizer(loss::L, θ, α, y_true)`
    - it should return the following : argmax_y (loss(y, y_true) + α ⟨θ,  y - y_true⟩)
    - useful to compute the gradient of an associated SSVM loss.

We provide some special cases with explicit formulae, such as:
- [`ZeroOneLoss`](@ref)

We also provide a generic wrapper [`GeneralStructuredLoss`](@ref) to build your own StructuredLossFunction.

"""
@traitdef IsStructuredLossFunction{L}

## Explicit versions

"""
    ZeroOneLoss

Zero-one loss for multiclass classification (`ZeroOneLoss(y, y_true)` equals 0 if `y == y_true`, else 1).
"""
struct ZeroOneLoss end

@traitimpl IsStructuredLossFunction{ZeroOneLoss}

function (::ZeroOneLoss)(y::AbstractArray, y_true::AbstractArray)
    return y != y_true ? 1.0 : 0.0
end

function compute_maximizer(
    zol::ZeroOneLoss, θ::AbstractVector, α::Real, y_true::AbstractVector
)
    base = 1:length(θ)
    Y = [base .== i for i in base]

    return Y[argmax(zol(Y[i], y_true) + α * θ[i] for i in base)] # ! there is probably a clever way to compute this
end

"""
    GeneralStructuredLoss{F1, F2}

```
delta_loss(y, y_true)
maximizer(θ, α, y_true) = argmax_y (delta_loss(y, y_true) + α ⟨θ,  y - y_true⟩)
```
"""
struct GeneralStructuredLoss{F1,F2}
    delta_loss::F1
    maximizer::F2
end

@traitimpl IsStructuredLossFunction{GeneralStructuredLoss}

function (gsl::GeneralStructuredLoss)(y::AbstractArray, y_true::AbstractArray)
    return gsl.delta_loss(y, y_true)
end

function compute_maximizer(
    structured_loss::GeneralStructuredLoss,
    θ::AbstractArray,
    α::Real,
    y_true::AbstractArray,
)
    return structured_loss.maximizer(θ, α, y_true)
end
