"""
    soft_argmax(z)

Soft argmax activation function `s(z) = (e^zᵢ / ∑ e^zⱼ)ᵢ`.

Corresponds to regularized prediction on the probability simplex with entropic penalty.
"""
function soft_argmax(z::AbstractVector; kwargs...)
    s = exp.(z) / sum(exp, z)
    return s
end

@traitimpl IsRegularized{typeof(soft_argmax)}

function compute_regularization(::typeof(soft_argmax), y::AbstractVector{R}) where {R<:Real}
    return isprobadist(y) ? negative_shannon_entropy(y) : typemax(R)
end
