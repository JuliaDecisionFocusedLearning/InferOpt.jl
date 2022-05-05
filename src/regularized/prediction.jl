## Abstract type

"""
    IsRegularizedPrediction{P}

Trait-based interface for regularized prediction functions of the form `θ ⟼ ŷ(θ) = argmax ⟨θ,y⟩ - Ω(y)`.

For a type `P` to comply with this interface, the following methods must exist:
- `(prediction::P)(θ)`
- `compute_regularization(prediction::P, y)`

We provide some special cases with explicit formulae, such as:
- [`one_hot_argmax`](@ref)
- [`softmax`](@ref)
- [`sparsemax`](@ref)
"""
@traitdef IsRegularizedPrediction{P}

@traitimpl IsRegularizedPrediction{typeof(one_hot_argmax)}
@traitimpl IsRegularizedPrediction{typeof(softmax)}
@traitimpl IsRegularizedPrediction{typeof(sparsemax)}

function compute_regularization(::typeof(one_hot_argmax), y::AbstractVector)
    return isprobadist(y) ? 0.0 : Inf
end

function compute_regularization(::typeof(softmax), y::AbstractVector)
    return isprobadist(y) ? -shannon_entropy(y) : Inf
end

function compute_regularization(::typeof(sparsemax), y::AbstractVector)
    return isprobadist(y) ? half_square_norm(y) : Inf
end
