"""
    ImitationLoss{L,R,P}

Generic imitation loss: max_y ℓ(y, y̅) + α θᵀ(y - y̅) - (Ω(y) - Ω(y̅)).

When `ℓ = 0`, this loss is equivalent to a [`FenchelYoungLoss`](@ref).
When `Ω = 0`, this loss is equivalent to the [`StructuredSVMLoss`](@ref).

# Fields
- `maximizer::P`: function that computes
    argmax_y ℓ(y, y̅) + α θᵀ(y - y̅) - (Ω(y) - Ω(y̅)), takes (θ, y_true, kwargs...)
    or (θ, t_true, kwargs...) as input
- `ℓ::L`: base loss, can either take (y, y_true) or (y, t_true) as input
- `Ω::R`: regularization, takes y as input
- `α::Float64`: default value of 1.0
"""
struct ImitationLoss{P,L,R}
    maximizer::P
    ℓ::L
    Ω::R
    α::Float64
end

function Base.show(io::IO, l::ImitationLoss)
    (; ℓ, Ω, maximizer, α) = l
    return print(io, "ImitationLoss($maximizer, $ℓ, $Ω, $α)")
end

"""
    ImitationLoss(maximizer[; ℓ=(y,t_true)->0.0, Ω=y->0.0, α=1.0])

Shorter constructor with defaults.
"""
function ImitationLoss(
    maximizer; ℓ=(y, t_true) -> 0.0, base_loss=nothing, Ω=y -> 0.0, omega=nothing, α=1.0
)
    if isnothing(base_loss) && isnothing(omega)
        return ImitationLoss(maximizer, ℓ, Ω, α)
    end
    if isnothing(base_loss)
        return ImitationLoss(maximizer, ℓ, omega, α)
    end
    if isnothing(omega)
        return ImitationLoss(maximizer, base_loss, Ω, α)
    end
    return ImitationLoss(maximizer, base_loss, omega, α)
end

getytrue(t_true) = error("not implemented")  # TODO: find a better function name ?
getytrue(t_true::NamedTuple) = t_true.y_true

# t_true must contain an `y_true` field
function prediction_and_loss(l::ImitationLoss, θ::AbstractArray{<:Real}, t_true; kwargs...)
    (; ℓ, Ω, maximizer, α) = l
    y_true = getytrue(t_true)
    ŷ = maximizer(θ, t_true; kwargs...)
    l = ℓ(ŷ, t_true) + α * (dot(θ, ŷ) - dot(θ, y_true)) + Ω(y_true) - Ω(ŷ)
    return ŷ, l
end

## Forward pass

function (l::ImitationLoss)(θ::AbstractArray{<:Real}, t_true; kwargs...)
    _, l = prediction_and_loss(l, θ, t_true; kwargs...)
    return l
end

## Backward pass

function ChainRulesCore.rrule(l::ImitationLoss, θ::AbstractArray{<:Real}, t_true; kwargs...)
    (; α) = l
    y_true = getytrue(t_true)
    ŷ, l = prediction_and_loss(l, θ, t_true; kwargs...)
    l_pullback(dl) = NoTangent(), dl .* α .* (ŷ .- y_true), NoTangent()
    return l, l_pullback
end
