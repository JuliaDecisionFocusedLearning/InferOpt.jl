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
    positive_part(x)

Compute `max(x,0)`.
"""
positive_part(x) = x >= 0 ? x : zero(x)

"""
    shannon_entropy(p)

Compute the Shannon entropy of a probability distribution: `H(p) = -∑ pᵢlog(pᵢ)`.
"""
function shannon_entropy(p::AbstractVector{R}) where {R<:Real}
    H = zero(R)
    for x in p
        if x > 0.0
            H -= x * log(x)
        end
    end
    return H
end

negative_shannon_entropy(p::AbstractVector{R}) where {R<:Real} = -shannon_entropy(p)

"""
    half_square_norm(x)

Compute the squared Euclidean norm of `x` and divide it by 2.
"""
function half_square_norm(x::AbstractVector{R}) where {R<:Real}
    return 0.5 * sum(abs2, x)
end

"""
    simplex_projection_and_support(z)

Compute the Euclidean projection `p` of `z` on the probability simplex (also called [`sparsemax`](@ref)), and the indicators `s` of its support.

See <https://arxiv.org/abs/1602.02068>.
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
