"""
$TYPEDEF

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
$TYPEDSIGNATURES

Compute L(θ, y_true).
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
    maximizer = get_maximizer(optimization_layer)
    l =
        (Ωy_true - objective_value(maximizer, θ, y_true; kwargs...)) -
        (Ωŷ - objective_value(maximizer, θ, ŷ; kwargs...))
    grad = apply_g(maximizer, ŷ; kwargs...) - apply_g(maximizer, y_true; kwargs...)
    return l, grad
end

function fenchel_young_loss_and_grad(
    fyl::FenchelYoungLoss{<:PerturbedOracle},
    θ::AbstractArray,
    y_true::AbstractArray;
    kwargs...,
)
    (; optimization_layer) = fyl
    maximizer = get_maximizer(optimization_layer)
    F, almost_ŷ = fenchel_young_F_and_first_part_of_grad(optimization_layer, θ; kwargs...)
    l = F - objective_value(maximizer, θ, y_true; kwargs...)
    g = almost_ŷ - apply_g(maximizer, y_true; kwargs...)
    return l, g
end

## Backward pass

function ChainRulesCore.rrule(
    fyl::FenchelYoungLoss, θ::AbstractArray, y_true::AbstractArray; kwargs...
)
    l, g = fenchel_young_loss_and_grad(fyl, θ, y_true; kwargs...)
    fyl_pullback(dl) = NoTangent(), dl * g, NoTangent()
    return l, fyl_pullback
end

## Specific overrides for perturbed layers

function fenchel_young_F_and_first_part_of_grad(
    perturbed::PerturbedOracle{<:AdditivePerturbation}, θ::AbstractArray; kwargs...
)
    (; reinforce) = perturbed
    maximizer = get_maximizer(perturbed)
    η_dist = empirical_predistribution(reinforce, θ)
    fk = FixKwargs(maximizer, kwargs)
    gk = Fix1Kwargs(apply_g, maximizer, kwargs)
    y_dist = map(fk, η_dist)
    F = mean(
        objective_value(maximizer, η, y; kwargs...) for
        (η, y) in zip(η_dist.atoms, y_dist.atoms)
    )
    ŷ = mean(gk, y_dist)
    return F, ŷ
end

function fenchel_young_F_and_first_part_of_grad(
    perturbed::PerturbedOracle{<:MultiplicativePerturbation}, θ::AbstractArray; kwargs...
)
    (; reinforce) = perturbed
    maximizer = get_maximizer(perturbed)
    η_dist = empirical_predistribution(reinforce, θ)
    fk = FixKwargs(reinforce.f, kwargs)
    gk = Fix1Kwargs(apply_g, maximizer, kwargs)
    y_dist = map(fk, η_dist)
    eZ_dist = map(Base.Fix2(./, θ), η_dist)
    F = mean(
        objective_value(maximizer, η, y; kwargs...) for
        (η, y) in zip(η_dist.atoms, y_dist.atoms)
    )
    almost_ŷ = mean(gk.(map(.*, eZ_dist.atoms, y_dist.atoms)))
    return F, almost_ŷ
end
