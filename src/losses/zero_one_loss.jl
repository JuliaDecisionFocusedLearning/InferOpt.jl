"""
    zero_one_loss(y, y_true)

0-1 loss for multiclass classification: `δ(y, y_true) = 0` if `y = y_true`, and `1` otherwise.
"""
function zero_one_loss(y::AbstractArray, y_true::AbstractArray)
    return y == y_true ? zero(eltype(y)) : one(eltype(y))
end

"""
    zero_one_loss_maximizer(y, y_true; α)

For `δ = zero_one_loss`, compute
```
argmax_y {δ(y, y_true) + α θᵀ(y - y_true)}
```
"""
function zero_one_loss_maximizer(
    θ::AbstractVector,
    y_true::AbstractVector{R},  # TODO: does it work with arrays?
    α;
    kwargs...,
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

"""
    ZeroOneStructuredSVMLoss

Implementation of the [`StructuredSVMLoss`](@ref) based on a 0-1 loss for multiclass classification.
"""
function ZeroOneStructuredSVMLoss(α=1)
    return StructuredSVMLoss(;
        aux_loss_maximizer=zero_one_loss_maximizer, δ=zero_one_loss, α=α
    )
end

"""
    ZeroOneStructuredSVMLoss(α)

Implementation of the [`ImitationLoss`](@ref) based on a 0-1 loss for multiclass classification with no regularization.
"""
function ZeroOneImitationLoss(α=1)
    return ImitationLoss(;
        δ=(y, t_true) -> zero_one_loss(y, get_y_true(t_true)),
        Ω=y -> 0,
        α=α,
        aux_loss_maximizer=(θ, t_true, α; kwargs...) ->
            zero_one_loss_maximizer(θ, get_y_true(t_true), α; kwargs...),
    )
end
