"""
    PerturbedNormal{F}

Differentiable normal perturbation of a black-box optimizer: `θ -> θ + εZ` where `Z ∼ N(0, 1)`.

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

function sample_perturbation(perturbed::PerturbedNormal, θ::AbstractArray)
    (; ε) = perturbed
    return θ .+ ε .* randn(size(θ))
end

function ChainRulesCore.rrule(perturbed::PerturbedNormal, θ::AbstractArray; kwargs...)
    (; maximizer, ε, M) = perturbed
    Z_samples = [randn(size(θ)) for _ in 1:M]
    y_samples = [maximizer(θ .+ ε .* Zs; kwargs...) for Zs in Z_samples]
    function perturbed_pullback(dy)
        vjp = inv(ε) * mean(dot(dy, ys) .* Zs for (Zs, ys) in zip(Z_samples, y_samples))
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
    y_samples = [maximizer(θ .+ ε .* Zs; kwargs...) for Zs in Z_samples]
    cost_samples = [cost(ys; kwargs...) for ys in y_samples]
    function perturbed_cost_pullback(dc)
        vjp = inv(ε) * mean((dc * cs) .* Zs for (cs, Zs) in zip(cost_samples, Z_samples))
        return NoTangent(), vjp, NoTangent()
    end
    return mean(cost_samples), perturbed_cost_pullback
end
