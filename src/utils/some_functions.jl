"""
$TYPEDSIGNATURES

Compute `max(x, 0)`.
"""
positive_part(x) = x >= zero(x) ? x : zero(x)

"""
$TYPEDSIGNATURES

Check whether `x ∈ [0,1]`.
"""
isproba(x::Real) = zero(x) <= x <= one(x)

"""
$TYPEDSIGNATURES

Check whether the elements of `p` are nonnegative and sum to 1.
"""
isprobadist(p::AbstractVector{R}) where {R<:Real} = all(isproba, p) && sum(p) ≈ one(R)

"""
$TYPEDSIGNATURES

Compute the squared Euclidean norm of `x` and divide it by 2.
"""
function half_square_norm(x::AbstractArray)
    return sum(abs2, x) / 2
end

"""
$TYPEDSIGNATURES

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

negative_shannon_entropy(p::AbstractVector) = -shannon_entropy(p)

"""
$TYPEDSIGNATURES

One-hot encoding of the argmax function.
"""
function one_hot_argmax(z::AbstractVector{R}; kwargs...) where {R<:Real}
    e = zeros(R, length(z))
    e[argmax(z)] = one(R)
    return e
end

"""
$TYPEDSIGNATURES

Compute the vector `r` such that `rᵢ` is the rank of `θᵢ` in `θ`.
"""
function ranking(θ::AbstractVector; rev::Bool=false, kwargs...)
    return invperm(sortperm(θ; rev=rev))
end

zero_regularization(y) = zero(eltype(y))
zero_gradient(y) = zero(y)

identity_kw(x; kwargs...) = identity(x)
eltype_kw(x; kwargs...) = eltype(x)
