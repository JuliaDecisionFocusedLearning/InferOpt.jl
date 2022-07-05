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
"""
@traitdef IsRegularized{P}

@traitfn function compute_regularization(predictor::P, y) where {P; IsRegularized{P}} end
