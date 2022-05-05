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
