## Abstract type

"""
    IsRegularizedPrediction{P}

Trait-based interface for regularized prediction functions of the form `θ ⟼ ŷ(θ) = argmax ⟨θ,y⟩ - Ω(y)`.

For a type `P` to comply with this interface, the following methods must exist:
- `(prediction::P)(θ)`
- `compute_regularization(prediction::P, y)`

We provide some special cases with explicit formulae, such as:
- [`one_hot_argmax`](@ref)
- [`soft_argmax`](@ref)
- [`sparse_argmax`](@ref)
"""
@traitdef IsRegularizedPrediction{P}

@traitimpl IsRegularizedPrediction{typeof(one_hot_argmax)}
@traitimpl IsRegularizedPrediction{typeof(soft_argmax)}
@traitimpl IsRegularizedPrediction{typeof(sparse_argmax)}

function compute_regularization(
    ::typeof(one_hot_argmax), y::AbstractVector{R}
) where {R<:Real}
    return isprobadist(y) ? zero(R) : typemax(R)
end

function compute_regularization(::typeof(soft_argmax), y::AbstractVector{R}) where {R<:Real}
    return isprobadist(y) ? -shannon_entropy(y) : typemax(R)
end

function compute_regularization(
    ::typeof(sparse_argmax), y::AbstractVector{R}
) where {R<:Real}
    return isprobadist(y) ? half_square_norm(y) : typemax(R)
end
