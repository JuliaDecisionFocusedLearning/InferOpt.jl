"""
    ZeroOneBaseLoss

0-1 loss for multiclass classification: `δ(y, y_true) = 0` if `y = y_true`, and `1` otherwise.
"""
struct ZeroOneBaseLoss end

@traitimpl IsBaseLoss{ZeroOneBaseLoss}

function (::ZeroOneBaseLoss)(y::AbstractArray, y_true::AbstractArray)
    return y == y_true ? zero(eltype(y)) : one(eltype(y))
end

function compute_maximizer(
    ::ZeroOneBaseLoss, θ::AbstractVector, α::Real, y_true::AbstractVector{R}
) where {R<:Real}
    i_true = findfirst(==(one(R)), y_true)
    i_θ = argmax(θ)
    y = zeros(R, size(y_true))
    if (i_true == i_θ) || (α * θ[i_true] > 1 + α * θ[i_θ])
        y[i_true] = one(R)
    else
        y[i_θ] = one(R)
    end
    return y
end
