"""
    ImitationLoss{L,R,P}

Generic imitation loss: max_y base_loss(y, t_true) + α θᵀ(y - y_true) - (Ω(y) - Ω(y_true))).

When `base_loss = 0`, this loss is equivalent to a [`FenchelYoungLoss`](@ref).
When `Ω = 0`, this loss is equivalent to the [`StructuredSVMLoss`](@ref).

# Fields
- `maximizer::P`: function that computes
    argmax_y base_loss(y, t_true) + α θᵀ(y - y_true) - (Ω(y) - Ω(y_true)), takes (θ, y_true, kwargs...)
    or (θ, t_true, kwargs...) as input
- `base_loss::L`: base loss, takes (y, t_true) as input
- `Ω::R`: regularization, takes y as input
- `α::Float64`: default value of 1.0

Note: by default, `t_true` is a named tuple with field `y_true`,
    but can be any data structure for which the [`get_y_true`](@ref) method is implemented.
"""
struct ImitationLoss{P,L,R}
    maximizer::P
    base_loss::L
    Ω::R
    α::Float64
end

function Base.show(io::IO, l::ImitationLoss)
    (; base_loss, Ω, maximizer, α) = l
    return print(io, "ImitationLoss($maximizer, $base_loss, $Ω, $α)")
end

"""
    ImitationLoss(maximizer[; base_loss=(y,t_true)->0.0, Ω=y->0.0, α=1.0])

Shorter constructor with defaults.
"""
function ImitationLoss(maximizer; base_loss=(y, t_true) -> 0.0, Ω=y -> 0.0, α=1.0)
    return ImitationLoss(maximizer, base_loss, Ω, α)
end

"""
    get_y_true(t_true::Any)

Retrieve `y_true` from `t_true`.
This method should be implemented when using a custom data structure for `t_true` other than a `NamedTuple`.
"""
function get_y_true end

"""
    get_y_true(t_true::NamedTuple)

Retrieve `y_true` from `t_true`. `t_true` must contain an `y_true` field.
"""
get_y_true(t_true::NamedTuple) = t_true.y_true

function prediction_and_loss(l::ImitationLoss, θ::AbstractArray, t_true; kwargs...)
    (; base_loss, Ω, maximizer, α) = l
    y_true = get_y_true(t_true)
    ŷ = maximizer(θ, t_true; kwargs...)
    l = base_loss(ŷ, t_true) + α * (dot(θ, ŷ) - dot(θ, y_true)) + Ω(y_true) - Ω(ŷ)
    return ŷ, l
end

## Forward pass

function (l::ImitationLoss)(θ::AbstractArray, t_true; kwargs...)
    _, l = prediction_and_loss(l, θ, t_true; kwargs...)
    return l
end

## Backward pass

function ChainRulesCore.rrule(l::ImitationLoss, θ::AbstractArray, t_true; kwargs...)
    (; α) = l
    y_true = get_y_true(t_true)
    ŷ, l = prediction_and_loss(l, θ, t_true; kwargs...)
    l_pullback(dl) = NoTangent(), dl .* α .* (ŷ .- y_true), NoTangent()
    return l, l_pullback
end
