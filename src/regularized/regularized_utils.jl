"""
    positive_part(x)

Compute `max(x,0)`.
"""
positive_part(x) = x >= zero(x) ? x : zero(x)

"""
    isproba(x)

Check whether `x ∈ [0,1]`.
"""
isproba(x::Real) = zero(x) <= x <= one(x)

"""
    isprobadist(p)

Check whether the elements of `p` are nonnegative and sum to 1.
"""
isprobadist(p::AbstractVector{R}) where {R<:Real} = all(isproba, p) && sum(p) ≈ one(R)

"""
    half_square_norm(x)

Compute the squared Euclidean norm of `x` and divide it by 2.
"""
function half_square_norm(x::AbstractArray{<:Real})
    return sum(abs2, x) / 2
end

"""
    shannon_entropy(p)

Compute the Shannon entropy of a probability distribution: `H(p) = -∑ pᵢlog(pᵢ)`.
"""
function shannon_entropy(p::AbstractVector{R}) where {R<:Real}
    H = zero(R)
    for x in p
        if x > zero(R)
            H -= x * log(x)
        end
    end
    return H
end

negative_shannon_entropy(p::AbstractVector{<:Real}) = -shannon_entropy(p)

"""
    one_hot_argmax(z)

One-hot encoding of the argmax function.
"""
function one_hot_argmax(z::AbstractVector{R}; kwargs...) where {R<:Real}
    e = zeros(R, length(z))
    e[argmax(z)] = one(R)
    return e
end

"""
    ranking(θ[; rev])

Compute the vector `r` such that `rᵢ` is the rank of `θᵢ` in `θ`.
"""
function ranking(θ::AbstractVector{<:Real}; rev::Bool=false, kwargs...)
    return invperm(sortperm(θ; rev=rev))
end

zero_regularization(y) = zero(eltype(y))
zero_gradient(y) = zero(y)

## Wrapper for linear maximizers to use them within Frank-Wolfe

"""
    LinearMaximizationOracle{F,K}

Wraps a linear maximizer as a `FrankWolfe.LinearMinimizationOracle` with a sign switch.

# Fields
- `maximizer::F`: black box linear maximizer
- `maximizer_kwargs::K`: keyword arguments passed to the maximizer whenever it is called
"""
struct LinearMaximizationOracle{F,K} <: LinearMinimizationOracle
    maximizer::F
    maximizer_kwargs::K
end

LinearMaximizationOracle(maximizer) = LinearMaximizationOracle(maximizer, NamedTuple())

"""
    FrankWolfe.compute_extreme_point(lmo_wrapper::LMOWrapper, direction)
"""
function FrankWolfe.compute_extreme_point(
    lmo::LinearMaximizationOracle, direction; kwargs...
)
    (; maximizer, maximizer_kwargs) = lmo
    opposite_direction = -direction
    v = maximizer(opposite_direction; maximizer_kwargs...)
    return v
end

"""
    DEFAULT_FRANK_WOLFE_KWARGS

Default configuration for the Frank-Wolfe wrapper.
"""
const DEFAULT_FRANK_WOLFE_KWARGS = (
    away_steps=true,
    epsilon=1e-4,
    lazy=true,
    line_search=FrankWolfe.Agnostic(),
    max_iteration=10,
    timeout=1.0,
    verbose=false,
)
