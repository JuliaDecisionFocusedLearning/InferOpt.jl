"""
    PerturbedNormal{F}

Differentiable normal perturbation of a black-box optimizer: the input undergoes `θ -> θ + εZ` where `Z ∼ N(0, 1)`.

# Fields
- `maximizer::F`: underlying argmax function
- `ε::Float64`: noise scaling parameter
- `M::Int`: number of noise samples for Monte-Carlo computations
"""
struct PerturbedNormal{F} <: AbstractPerturbed{F}
    maximizer::F
    ε::Float64
    M::Int
end

PerturbedNormal(maximizer; ε=1.0, M=2) = PerturbedNormal(maximizer, float(ε), M)

function (perturbed::PerturbedNormal)(θ::AbstractArray, Z::AbstractArray; kwargs...)
    (; maximizer, ε) = perturbed
    return maximizer(θ .+ ε .* Z; kwargs...)
end

## Fenchel-Young loss

function compute_y_and_F(
    perturbed::PerturbedNormal, θ::AbstractArray, Z::AbstractArray; kwargs...
)
    (; maximizer, ε) = perturbed
    y = maximizer(θ .+ ε .* Z; kwargs...)
    F = dot(θ .+ ε .* Z, y)
    return y, F
end

## Backward pass

function ChainRulesCore.rrule(perturbed::PerturbedNormal, θ::AbstractArray; kwargs...)
    (; maximizer, ε, M) = perturbed
    Z_samples = [randn(size(θ)) for _ in 1:M]
    y_samples = [maximizer(θ .+ ε .* Z; kwargs...) for Z in Z_samples]
    function perturbed_pullback(dy)
        vjp = inv(ε) * mean(dot(dy, y) .* Z for (Z, y) in zip(Z_samples, y_samples))
        return NoTangent(), vjp
    end
    return mean(y_samples), perturbed_pullback
end

function ChainRulesCore.rrule(
    perturbed_cost::PerturbedCost{F,P}, θ::AbstractArray; kwargs...
) where {F,P<:PerturbedNormal{F}}
    (; perturbed, cost) = perturbed_cost
    (; maximizer, ε, M) = perturbed
    Z_samples = [randn(size(θ)) for _ in 1:M]
    y_samples = [maximizer(θ .+ ε .* Z; kwargs...) for Z in Z_samples]
    cost_samples = [cost(ys; kwargs...) for ys in y_samples]
    function perturbed_cost_pullback(dc)
        vjp = inv(ε) * mean((dc * c) .* Z for (Z, c) in zip(Z_samples, cost_samples))
        return NoTangent(), vjp, NoTangent()
    end
    return mean(cost_samples), perturbed_cost_pullback
end
