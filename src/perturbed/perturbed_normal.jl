"""
    Perturbed{F}

Differentiable normal perturbation of a black-box optimizer.

# Fields
- `maximizer::F`: underlying argmax function
- `ε::Float64`: noise scaling parameter
- `M::Int`: number of noise samples for Monte-Carlo computations
"""
struct Perturbed{F} <: AbstractPerturbed
    maximizer::F
    ε::Float64
    M::Int
end

Perturbed(maximizer; ε=1.0, M=2) = Perturbed(maximizer, float(ε), M)

function (perturbed::Perturbed)(θ::AbstractArray; kwargs...)
    (; maximizer, ε, M) = perturbed
    d = size(θ)
    y_samples = [maximizer(θ + ε * randn(d); kwargs...) for _ in 1:M]
    y_mean = mean(y_samples)
    return y_mean
end

function compute_y_and_Fθ(perturbed::Perturbed, θ::AbstractArray; kwargs...)
    (; maximizer, ε, M) = perturbed
    d = size(θ)
    θ_samples = [θ + ε * randn(d) for _ in 1:M]
    y_samples = [maximizer(θ_sample; kwargs...) for θ_sample in θ_samples]
    F_θ_samples = [dot(θ_sample, y) for (θ_sample, y) in zip(θ_samples, y_samples)]
    y_mean = mean(y_samples)
    Fθ_mean = mean(F_θ_samples)  # useful for computing Fenchel-Young loss
    return y_mean, Fθ_mean
end

function ChainRulesCore.rrule(perturbed::Perturbed, θ::AbstractArray; kwargs...)
    (; maximizer, ε, M) = perturbed
    d = size(θ)
    Z_samples = [randn(d) for m in 1:M]
    y_samples = [maximizer(θ + ε * Z; kwargs...) for Z in Z_samples]
    y_mean = mean(y_samples)

    function perturbed_pullback(dy)
        vjp = (1 / ε) * mean(dot(dy, y_samples[m]) * Z_samples[m] for m in 1:M)
        return NoTangent(), vjp
    end

    return y_mean, perturbed_pullback
end

"""
    PerturbedCost{F,C}

Composition of a differentiable perturbed black-box optimizer with an arbitrary cost function.

Designed for direct regret minimization (learning by experience).

# Fields
- `maximizer::F`: underlying argmax function
- `ε::Float64`: noise scaling parameter
- `M::Int`: number of noise samples for Monte-Carlo computations
- `cost::C`: a real-valued function taking a vector `y` and an instance as inputs

# See also
- [`Perturbed`](@ref)
"""
struct PerturbedCost{F,C}
    maximizer::F
    cost::C
    ε::Float64
    M::Int
end

PerturbedCost(maximizer, cost; ε=1.0, M=2) = PerturbedCost(maximizer, cost, ε, M)

function (perturbed_cost::PerturbedCost)(θ::AbstractArray; kwargs...)
    (; maximizer, cost, ε, M) = perturbed_cost
    d = size(θ)
    y_samples = [maximizer(θ + ε * randn(d); kwargs...) for m in 1:M]
    costs = [cost(y; kwargs...) for y in y_samples]
    return mean(costs)
end

function ChainRulesCore.rrule(perturbed_cost::PerturbedCost, θ::AbstractArray; kwargs...)
    (; maximizer, cost, ε, M) = perturbed_cost
    d = size(θ)
    Z_samples = [randn(d) for _ in 1:M]
    y_samples = [maximizer(θ + ε * Z; kwargs...) for Z in Z_samples]
    costs = [cost(y; kwargs...) for y in y_samples]

    function perturbed_cost_pullback(dc)
        vjp = (dc / ε) * mean(costs[m] * Z_samples[m] for m in 1:M)
        return NoTangent(), vjp, NoTangent()
    end

    return mean(costs), perturbed_cost_pullback
end