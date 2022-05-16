lognormal_gradlogpdf(x::Real, μ::Real, σ::Real) = (log(x) - μ - one(x)) / (x * σ^2)

"""
    PerturbedLogNormal{F}

Differentiable log-normal perturbation of a black-box optimizer.

# Fields
- `maximizer::F`: underlying argmax function
- `ε::Float64`: noise scaling parameter
- `M::Int`: number of noise samples for Monte-Carlo computations
"""
struct PerturbedLogNormal{F} <: AbstractPerturbed
    maximizer::F
    ε::Float64
    M::Int
end

PerturbedLogNormal(maximizer; ε=1.0, M=2) = PerturbedLogNormal(maximizer, float(ε), M)

function (perturbed::PerturbedLogNormal)(θ::AbstractArray; kwargs...)
    (; maximizer, ε, M) = perturbed
    d = size(θ)
    θ_samples = [exp.(log.(θ) + ε * randn(d) - ε^2 / 2) for _ in 1:M]
    y_samples = [maximizer(θ_sample; kwargs...) for θ_sample in θ_samples]
    y_mean = mean(y_samples)
    return y_mean
end

function compute_y_and_Fθ(perturbed::PerturbedLogNormal, θ::AbstractArray; kwargs...)
    (; maximizer, ε, M) = perturbed
    d = size(θ)
    θ_samples = [exp.(log.(θ) + ε * randn(d) - ε^2 / 2) for _ in 1:M]
    y_samples = [maximizer(θ_sample; kwargs...) for θ_sample in θ_samples]
    F_θ_samples = [dot(θ_sample, y) for (θ_sample, y) in zip(θ_samples, y_samples)]
    y_mean = mean(y_samples)
    Fθ_mean = mean(F_θ_samples)  # useful for computing Fenchel-Young loss
    return y_mean, Fθ_mean
end
