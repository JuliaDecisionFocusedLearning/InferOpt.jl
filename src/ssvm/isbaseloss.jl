"""
    IsBaseLoss{L}

Trait-based interface for loss functions `δ(y, y_true)`, which are the base of the more complex `StructuredSVMLoss`.

For `δ::L` to comply with this interface, the following methods must exist:
- `(δ)(y, y_true)`
- [`compute_maximizer(δ, θ, α, y_true)`](@ref)

# Available implementations
- [`ZeroOneBaseLoss`](@ref)
"""
@traitdef IsBaseLoss{L}

"""
    compute_maximizer(δ, θ, α, y_true)

Compute `argmax_y {δ(y, y_true) + α θᵀ(y - y_true)}` to deduce the gradient of a `StructuredSVMLoss`.
"""
function compute_maximizer end
