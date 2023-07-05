"""
    SparseArgmax <: AbstractRegularized

Compute the Euclidean projection of the vector `z` onto the probability simplex.

Corresponds to regularized prediction on the probability simplex with square norm penalty.
"""
struct SparseArgmax <: AbstractRegularized end

(::SparseArgmax)(z) = sparse_argmax(z)
compute_regularization(::SparseArgmax, y) = sparse_argmax_regularization(y)

function sparse_argmax(z::AbstractVector; kwargs...)
    p, _ = simplex_projection_and_support(z)
    return p
end

function sparse_argmax_regularization(y::AbstractVector)
    return isprobadist(y) ? half_square_norm(y) : typemax(eltype(y))
end

"""
    simplex_projection_and_support(z)

Compute the Euclidean projection `p` of `z` on the probability simplex (also called `sparse_argmax`), and the indicators `s` of its support.

Reference: <https://arxiv.org/abs/1602.02068>.
"""
function simplex_projection_and_support(z::AbstractVector)
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

function ChainRulesCore.rrule(::typeof(sparse_argmax), z::AbstractVector)
    p, s = simplex_projection_and_support(z)
    S = sum(s)
    function sparse_argmax_pullback(dp)
        vjp = s .* (dp .- (dp's) / S)
        return (NoTangent(), vjp)
    end
    return p, sparse_argmax_pullback
end
