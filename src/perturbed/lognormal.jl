
"""
    PerturbedLogNormal{F}

Differentiable log-normal perturbation of a black-box optimizer.

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

function sample_perturbation(perturbed::PerturbedLogNormal, θ::AbstractArray)
    (; ε) = perturbed
    return sign.(θ) .* exp.(log.(abs.(θ)) .- ε^2 / 2 .+ ε .* randn(size(θ)))
end

function gradlogpdf_perturbation(
    perturbed::PerturbedLogNormal, θs::AbstractArray, θ::AbstractArray
)
    error("Not implemented")
end

difflogpdf_lognormal(x::Real, μ::Real, σ::Real) = -inv(x) - (log(x) - μ) / (x * σ^2)

function compute_y_and_Fθ(perturbed::PerturbedLogNormal, θ::AbstractArray; kwargs...)
    (; maximizer, M) = perturbed
    θ_samples = [sample_perturbation(perturbed, θ) for _ in 1:M]
    y_samples = [maximizer(θs; kwargs...) for θs in θ_samples]
    return mean(y_samples), NaN
end
