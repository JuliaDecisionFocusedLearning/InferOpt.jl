"""
    IsRegularized{P}

Trait-based interface for regularized prediction functions `ŷ(θ) = argmax {θᵀy - Ω(y)}`.

For `predictor::P` to comply with this interface, the following methods must exist:
- `(predictor)(θ)`
- `compute_regularization(predictor, y)`

# Available implementations
- [`one_hot_argmax`](@ref)
- [`soft_argmax`](@ref)
- [`sparse_argmax`](@ref)
- [`RegularizedGeneric`](@ref)
"""
@traitdef IsRegularized{P}

"""
    compute_regularization(predictor::P, y)

Compute the convex regularization function `Ω(y)`.
"""
function compute_regularization end
