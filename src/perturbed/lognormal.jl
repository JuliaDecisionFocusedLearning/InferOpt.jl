"""
    PerturbedLogNormal{F}

Differentiable log-normal perturbation of a black-box optimizer: `θ -> exp[εZ - ε²/2] * θ` where `Z ∼ N(0, 1)`.

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
    return exp.(ε .* randn(size(θ)) .- ε^2) .* θ
end

difflogpdf_lognormal(x::Real, μ::Real, σ::Real) = -inv(x) - (log(x) - μ) / (x * σ^2)
