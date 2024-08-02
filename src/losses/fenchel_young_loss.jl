"""
    FenchelYoungLoss <: AbstractLossLayer

Fenchel-Young loss associated with a given optimization layer.
```
L(θ, y_true) = (Ω(y_true) - θᵀy_true) - (Ω(ŷ) - θᵀŷ)
```

Reference: <https://arxiv.org/abs/1901.02324>

# Fields

- `optimization_layer::AbstractOptimizationLayer`: optimization layer that can be formulated as `ŷ(θ) = argmax {θᵀy - Ω(y)}` (either regularized or perturbed)
"""
struct FenchelYoungLoss{O<:AbstractOptimizationLayer} <: AbstractLossLayer
    optimization_layer::O
end

function Base.show(io::IO, fyl::FenchelYoungLoss)
    (; optimization_layer) = fyl
    return print(io, "FenchelYoungLoss($optimization_layer)")
end

## Forward pass

"""
    (fyl::FenchelYoungLoss)(θ, y_true[; kwargs...])
"""
function (fyl::FenchelYoungLoss)(θ::AbstractArray, y_true::AbstractArray; kwargs...)
    l, _ = fenchel_young_loss_and_grad(fyl, θ, y_true; kwargs...)
    return l
end

function fenchel_young_loss_and_grad(
    fyl::FenchelYoungLoss{O}, θ::AbstractArray, y_true::AbstractArray; kwargs...
) where {O<:AbstractRegularized}
    (; optimization_layer) = fyl
    ŷ = optimization_layer(θ; kwargs...)
    Ωy_true = compute_regularization(optimization_layer, y_true)
    Ωŷ = compute_regularization(optimization_layer, ŷ)
    l = (Ωy_true - dot(θ, y_true)) - (Ωŷ - dot(θ, ŷ))
    g = ŷ - y_true
    return l, g
end

function fenchel_young_loss_and_grad(
    fyl::FenchelYoungLoss{O}, θ::AbstractArray, y_true::AbstractArray; kwargs...
) where {O<:AbstractRegularizedGeneralizedMaximizer}
    (; optimization_layer) = fyl
    ŷ = optimization_layer(θ; kwargs...)
    Ωy_true = compute_regularization(optimization_layer, y_true)
    Ωŷ = compute_regularization(optimization_layer, ŷ)
    maximizer = get_maximizer(optimization_layer)
    l =
        (Ωy_true - objective_value(maximizer, θ, y_true; kwargs...)) -
        (Ωŷ - objective_value(maximizer, θ, ŷ; kwargs...))
    g = maximizer.g(ŷ; kwargs...) - maximizer.g(y_true; kwargs...)
    return l, g
end

function fenchel_young_loss_and_grad(
    fyl::FenchelYoungLoss{<:Perturbed}, θ::AbstractArray, y_true::AbstractArray; kwargs...
)
    (; optimization_layer) = fyl
    maximizer = get_maximizer(optimization_layer)
    F, almost_ŷ = fenchel_young_F_and_first_part_of_grad(optimization_layer, θ; kwargs...)
    l = F - objective_value(maximizer, θ, y_true; kwargs...) # dot(θ, y_true)
    g = almost_ŷ - apply_g(maximizer, y_true; kwargs...)
    return l, g
end

# function fenchel_young_loss_and_grad(
#     fyl::FenchelYoungLoss{<:Perturbed{<:GeneralizedMaximizer}},
#     θ::AbstractArray,
#     y_true::AbstractArray;
#     kwargs...,
# )
#     (; optimization_layer) = fyl
#     F, almost_g_of_ŷ = fenchel_young_F_and_first_part_of_grad(
#         optimization_layer, θ; kwargs...
#     )
#     l = F - objective_value(optimization_layer.reinforce.f, θ, y_true; kwargs...)
#     g = almost_g_of_ŷ - optimization_layer.reinforce.f.g(y_true; kwargs...)
#     return l, g
# end

## Backward pass

function ChainRulesCore.rrule(
    fyl::FenchelYoungLoss, θ::AbstractArray, y_true::AbstractArray; kwargs...
)
    l, g = fenchel_young_loss_and_grad(fyl, θ, y_true; kwargs...)
    fyl_pullback(dl) = NoTangent(), dl * g, NoTangent()
    return l, fyl_pullback
end

## Specific overrides for perturbed layers

# function compute_F_and_y_samples(
#     perturbed::AbstractPerturbed{O,false},
#     θ::AbstractArray,
#     Z_samples::Vector{<:AbstractArray};
#     kwargs...,
# ) where {O}
#     F_and_y_samples = [
#         fenchel_young_F_and_first_part_of_grad(perturbed, θ, Z; kwargs...) for
#         Z in Z_samples
#     ]
#     return F_and_y_samples
# end

# function compute_F_and_y_samples(
#     perturbed::AbstractPerturbed{O,true},
#     θ::AbstractArray,
#     Z_samples::Vector{<:AbstractArray};
#     kwargs...,
# ) where {O}
#     return ThreadsX.map(
#         Z -> fenchel_young_F_and_first_part_of_grad(perturbed, θ, Z; kwargs...), Z_samples
#     )
# end

function fenchel_young_F_and_first_part_of_grad(
    perturbed::Perturbed{F,<:AdditivePerturbation}, θ::AbstractArray; kwargs...
) where {F}
    (; reinforce) = perturbed
    maximizer = get_maximizer(perturbed)
    η_dist = empirical_predistribution(reinforce, θ)
    fk = FixKwargs(maximizer, kwargs)
    gk = FixKwargs((y; kwargs...) -> apply_g(maximizer, y; kwargs...), kwargs)
    y_dist = map(fk, η_dist)
    return mean(
        objective_value(maximizer, η, y; kwargs...) for
        (η, y) in zip(η_dist.atoms, y_dist.atoms)
    ),
    mean(gk, y_dist)
end

# function fenchel_young_F_and_first_part_of_grad(
#     perturbed::Perturbed{<:GeneralizedMaximizer,<:AdditivePerturbation},
#     θ::AbstractArray;
#     kwargs...,
# )
#     (; reinforce) = perturbed
#     η_dist = empirical_predistribution(reinforce, θ)
#     fk = FixKwargs(reinforce.f, kwargs)
#     gk = FixKwargs(reinforce.f.g, kwargs)
#     y_dist = map(fk, η_dist)
#     return mean(
#         objective_value(reinforce.f, η, y; kwargs...) for
#         (η, y) in zip(η_dist.atoms, y_dist.atoms)
#     ),
#     mean(gk, y_dist)
# end

function fenchel_young_F_and_first_part_of_grad(
    perturbed::Perturbed{F,<:MultiplicativePerturbation}, θ::AbstractArray; kwargs...
) where {F}
    (; reinforce) = perturbed
    maximizer = get_maximizer(perturbed)
    η_dist = empirical_predistribution(reinforce, θ)
    fk = FixKwargs(reinforce.f, kwargs)
    gk = FixKwargs((y; kwargs...) -> apply_g(maximizer, y; kwargs...), kwargs)
    y_dist = map(fk, η_dist)
    eZ_dist = map(Base.Fix2(./, θ), η_dist)
    return mean(
        objective_value(maximizer, η, y; kwargs...) for
        (η, y) in zip(η_dist.atoms, y_dist.atoms)
    ),
    mean(gk.(map(.*, eZ_dist.atoms, y_dist.atoms)))
end

# function fenchel_young_F_and_first_part_of_grad(
#     perturbed::Perturbed{<:GeneralizedMaximizer,<:MultiplicativePerturbation},
#     θ::AbstractArray;
#     kwargs...,
# )
#     (; reinforce) = perturbed
#     η_dist = empirical_predistribution(reinforce, θ)
#     eZ_dist = map(Base.Fix2(./, θ), η_dist)
#     fk = FixKwargs(reinforce.f, kwargs)
#     gk = FixKwargs(reinforce.f.g, kwargs)
#     y_dist = map(fk, η_dist)
#     return mean(
#         objective_value(reinforce.f, η, y; kwargs...) for
#         (η, y) in zip(η_dist.atoms, y_dist.atoms)
#     ),
#     mean(gk.(map(.*, eZ_dist.atoms, y_dist.atoms)))
# end

# function fenchel_young_F_and_first_part_of_grad(
#     perturbed::PerturbedMultiplicative{P,G,O}, θ::AbstractArray, Z::AbstractArray; kwargs...
# ) where {P,G,O<:GeneralizedMaximizer}
#     (; oracle, ε) = perturbed
#     eZ = exp.(ε .* Z .- ε^2)
#     η = θ .* eZ
#     y = oracle(η; kwargs...)
#     F = objective_value(oracle, η, y; kwargs...)
#     y_scaled = y .* eZ
#     return F, oracle.g(y_scaled; kwargs...)
# end
