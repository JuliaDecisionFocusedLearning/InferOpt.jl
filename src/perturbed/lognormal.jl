"""
    PerturbedLogNormal{F}

Differentiable log-normal perturbation of a black-box optimizer: the input undergoes `θ -> exp[εZ - ε²/2] * θ` where `Z ∼ N(0, 1)`.

# Fields
- `maximizer::F`: underlying argmax function
- `ε::Float64`: noise scaling parameter
- `M::Int`: number of noise samples for Monte-Carlo computations
"""
struct PerturbedLogNormal{F} <: AbstractPerturbed{F}
    maximizer::F
    ε::Float64
    M::Int
end

PerturbedLogNormal(maximizer; ε=1.0, M=2) = PerturbedLogNormal(maximizer, float(ε), M)

function (perturbed::PerturbedLogNormal)(θ::AbstractArray, Z::AbstractArray; kwargs...)
    (; maximizer, ε) = perturbed
    eZ = exp.(ε .* Z .- ε^2)
    return eZ .* maximizer(eZ .* θ; kwargs...)
end

## Fenchel-Young loss

function compute_y_and_F(
    perturbed::AbstractPerturbed, θ::AbstractArray, Z::AbstractArray; kwargs...
)
    (; maximizer, ε) = perturbed
    eZ = exp.(ε .* Z .- ε^2)
    y_unscaled = maximizer(eZ .* θ; kwargs...)
    y = eZ .* y_unscaled
    F = dot(eZ .* θ, y_unscaled)
    return y, F
end

## Backward pass

function ChainRulesCore.rrule(perturbed::PerturbedLogNormal, θ::AbstractArray; kwargs...)
    (; maximizer, ε, M) = perturbed
    Z_samples = [randn(size(θ)) for _ in 1:M]
    y_samples = [maximizer(θ .* exp.(ε .* Z .- ε^2) ; kwargs...) for Z in Z_samples]
    function perturbed_pullback(dy)
        vjp = inv(ε) .* mean(dot(dy, y) .* Z for (Z, y) in zip(Z_samples, y_samples)) ./θ
        return NoTangent(), vjp
    end
    return mean(y_samples), perturbed_pullback
end

function ChainRulesCore.rrule(
    perturbed_cost::PerturbedCost{F,P}, θ::AbstractArray; kwargs...
) where {F,P<:PerturbedLogNormal{F}}
    (; perturbed, cost) = perturbed_cost
    (; maximizer, ε, M) = perturbed
    Z_samples = [randn(size(θ)) for _ in 1:M]
    y_samples = [maximizer(θ .* exp.(ε .* Z .- ε^2) ; kwargs...) for Z in Z_samples]
    cost_samples = [cost(ys; kwargs...) for ys in y_samples]
    function perturbed_cost_pullback(dc)
        vjp = inv(ε) .* mean((dc * c) .* Z for (Z, c) in zip(Z_samples, cost_samples)) ./θ
        return NoTangent(), vjp, NoTangent()
    end
    return mean(cost_samples), perturbed_cost_pullback
end