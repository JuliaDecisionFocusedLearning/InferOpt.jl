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

## Explicit versions

"""
    one_hot_argmax(θ)

One-hot encoding of the argmax function.

Corresponds to regularized prediction on the probability simplex with zero penalty.
"""
function one_hot_argmax(θ::AbstractVector)
    e = zeros(Float64, length(θ))
    e[argmax(θ)] = 1
    return e
end

@traitimpl IsRegularizedPrediction{typeof(one_hot_argmax)}

function compute_regularization(::typeof(one_hot_argmax), y::AbstractVector)
    if isprobadist(y)
        return 0.0
    else
        return Inf
    end
end

"""
    softmax(θ)

Softmax function `s(θ) = (e^θᵢ / ∑ e^θⱼ)ᵢ`.

Corresponds to regularized prediction on the probability simplex with entropic penalty.
"""
function softmax(θ::AbstractVector)
    e = exp.(θ)
    return e ./ sum(e)
end

@traitimpl IsRegularizedPrediction{typeof(softmax)}

function compute_regularization(::typeof(softmax), y::AbstractVector)
    if isprobadist(y)
        return -shannon_entropy(y)
    else
        return Inf
    end
end

"""
    sparsemax(z)

Project the vector `z` onto the probability simplex `Δ` in time `O(d log d)`.

Implementation and chain rule from <https://arxiv.org/abs/1602.02068>.

Corresponds to regularized prediction on the probability simplex with square norm penalty.
"""
function sparsemax(z::AbstractVector)
    p, _ = simplex_projection_and_support(z)
    return p
end

@traitimpl IsRegularizedPrediction{typeof(sparsemax)}

function compute_regularization(::typeof(sparsemax), y::AbstractVector)
    if isprobadist(y)
        return 0.5 * sum(abs2, y)
    else
        return Inf
    end
end

function ChainRulesCore.rrule(::typeof(sparsemax), z::AbstractVector)
    p, s = simplex_projection_and_support(z)
    S = sum(s)
    function sparsemax_pullback(dp)
        vjp = s .* (dp .- (dp's) / S)
        return (NoTangent(), vjp)
    end
    return p, sparsemax_pullback
end
