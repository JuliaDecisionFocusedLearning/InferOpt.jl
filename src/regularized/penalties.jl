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
    half_square_norm(x)

Compute the squared Euclidean norm of `x` and divide it by 2.
"""
function half_square_norm(x::AbstractArray{<:Real})
    return 0.5 * sum(abs2, x)
end
