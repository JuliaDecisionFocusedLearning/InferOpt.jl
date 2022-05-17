"""
    PerturbedNormal{F}

Differentiable normal perturbation of a black-box optimizer: `θ -> θ + N(0, ε²)`.

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
    return θ .+ perturbed.ε .* randn(size(θ))
end

function gradlogpdf_perturbation(
    perturbed::PerturbedNormal, θs::AbstractArray, θ::AbstractArray
)
    return (θ .- θs) ./ (perturbed.ε^2)
end
