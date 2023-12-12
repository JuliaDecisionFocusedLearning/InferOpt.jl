"""
$TYPEDEF

Generic imitation loss of the form
```
L(θ, t_true) = max_y {δ(y, t_true) + α θᵀ(y - y_true) - (Ω(y) - Ω(y_true))}
```

- When `δ` is zero, this is equivalent to a [`FenchelYoungLoss`](@ref).
- When `Ω` is zero, this is equivalent to a [`StructuredSVMLoss`](@ref).

Note: by default, `t_true` is a named tuple with field `y_true`, but it can be any data structure for which the [`get_y_true`](@ref) method is implemented.

# Fields
$TYPEDFIELDS
"""
struct ImitationLoss{M,L,R} <: AbstractLossLayer
    "function of `(θ, t_true, α)` that computes the argmax in the problem above"
    aux_loss_maximizer::M
    "base loss function"
    δ::L
    "regularization function"
    Ω::R
    "hyperparameter with a default value of 1.0"
    α::Float64
end

function Base.show(io::IO, l::ImitationLoss)
    (; aux_loss_maximizer, δ, Ω, α) = l
    return print(io, "ImitationLoss($aux_loss_maximizer, $δ, $Ω, $α)")
end

"""
    ImitationLoss(; aux_loss_maximizer, δ, Ω, α=1.0)

Explicit constructor with keyword arguments.
"""
function ImitationLoss(; aux_loss_maximizer, δ, Ω, α=1.0)
    return ImitationLoss(aux_loss_maximizer, δ, Ω, float(α))
end

"""
    $FUNCTIONNAME(t_true::Any)

Retrieve `y_true` from `t_true`.

This method should be implemented when using a custom data structure for `t_true` other than a `NamedTuple`.
"""
function get_y_true end

"""
$TYPEDSIGNATURES

Retrieve `y_true` from `t_true`. `t_true` must contain an `y_true` field.
"""
get_y_true(t_true::NamedTuple) = t_true.y_true

function prediction_and_loss(l::ImitationLoss, θ::AbstractArray, t_true; kwargs...)
    (; aux_loss_maximizer, δ, Ω, α) = l
    y_true = get_y_true(t_true)
    ŷ = aux_loss_maximizer(θ, t_true, α; kwargs...)
    l = δ(ŷ, t_true) + α * (dot(θ, ŷ) - dot(θ, y_true)) + Ω(y_true) - Ω(ŷ)
    return ŷ, l
end

## Forward pass

"""
    (il::ImitationLoss)(θ, t_true; kwargs...)
"""
function (il::ImitationLoss)(θ::AbstractArray, t_true; kwargs...)
    _, l = prediction_and_loss(il, θ, t_true; kwargs...)
    return l
end

## Backward pass

function ChainRulesCore.rrule(il::ImitationLoss, θ::AbstractArray, t_true; kwargs...)
    (; α) = il
    y_true = get_y_true(t_true)
    ŷ, l = prediction_and_loss(il, θ, t_true; kwargs...)
    il_pullback(dl) = NoTangent(), dl .* α .* (ŷ .- y_true), NoTangent()
    return l, il_pullback
end
