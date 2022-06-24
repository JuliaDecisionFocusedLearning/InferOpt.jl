"""
    sparse_argmax(z)

Compute the Euclidean projection of the vector `z` onto the probability simplex.

Corresponds to regularized prediction on the probability simplex with square norm penalty.
"""
function sparse_argmax(z::AbstractVector{<:Real}; kwargs...)
    p, _ = simplex_projection_and_support(z)
    return p
end

@traitimpl IsRegularized{typeof(sparse_argmax)}

function compute_regularization(
    ::typeof(sparse_argmax), y::AbstractVector{R}
) where {R<:Real}
    return isprobadist(y) ? half_square_norm(y) : typemax(R)
end

"""
    simplex_projection_and_support(z)

Compute the Euclidean projection `p` of `z` on the probability simplex (also called [`sparse_argmax`](@ref)), and the indicators `s` of its support.

Reference: <https://arxiv.org/abs/1602.02068>.
"""
function simplex_projection_and_support(z::AbstractVector{<:Real})
    d = length(z)
    z_sorted = sort(z; rev=true)
    z_sorted_cumsum = cumsum(z_sorted)
    k = maximum(j for j in 1:d if (1 + j * z_sorted[j]) > z_sorted_cumsum[j])
    τ = (z_sorted_cumsum[k] - 1) / k
    p = z .- τ
    p .= positive_part.(p)
    s = [Int(p[i] > eps()) for i in 1:d]
    return p, s
end

function ChainRulesCore.rrule(::typeof(sparse_argmax), z::AbstractVector{<:Real})
    p, s = simplex_projection_and_support(z)
    S = sum(s)
    function sparse_argmax_pullback(dp)
        vjp = s .* (dp .- (dp's) / S)
        return (NoTangent(), vjp)
    end
    return p, sparse_argmax_pullback
end
