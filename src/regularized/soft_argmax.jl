"""
$TYPEDEF

Soft argmax activation function `s(z) = (e^zᵢ / ∑ e^zⱼ)ᵢ`.

Corresponds to regularized prediction on the probability simplex with entropic penalty.
"""
struct SoftArgmax <: AbstractRegularized end

(::SoftArgmax)(z::AbstractVector; kwargs...) = soft_argmax(z)
compute_regularization(::SoftArgmax, y) = soft_argmax_regularization(y)

"""
$TYPEDSIGNATURES
"""
function soft_argmax(z::AbstractVector)
    s = exp.(z)
    return s ./ sum(s)
end

function soft_argmax_regularization(y::AbstractVector)
    return isprobadist(y) ? negative_shannon_entropy(y) : typemax(eltype(y))
end
