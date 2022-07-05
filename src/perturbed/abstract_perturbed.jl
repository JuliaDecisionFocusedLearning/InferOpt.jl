"""
    AbstractPerturbed

Differentiable perturbation of a black-box optimizer.

# Available subtypes
- [`PerturbedAdditive{F}`](@ref)
- [`PerturbedMultiplicative{F}`](@ref)

# Required fields
- `rng::AbstractRNG`: random number generator
- `seed::Union{Nothing,Int}`: random seed
- `nb_samples::Int`: number of random samples for Monte-Carlo computations
"""
abstract type AbstractPerturbed{F} end

function sample_perturbations(perturbed::AbstractPerturbed, θ::AbstractArray{<:Real})
    (; rng, seed, nb_samples) = perturbed
    seed!(rng, seed)
    Z_samples = [randn(rng, size(θ)) for _ in 1:nb_samples]
    return Z_samples
end

"""
    perturb_and_optimize(perturbed, θ, Z; kwargs...)
"""
function perturb_and_optimize(
    perturbed::AbstractPerturbed,
    θ::AbstractArray{<:Real},
    Z::AbstractArray{<:Real};
    kwargs...,
)
    return error("Not implemented")
end

function compute_probability_distribution(
    perturbed::AbstractPerturbed,
    θ::AbstractArray{<:Real},
    Z_samples::Vector{<:AbstractArray{<:Real}};
    kwargs...,
)
    atoms = [perturb_and_optimize(perturbed, θ, Z; kwargs...) for Z in Z_samples]
    weights = ones(length(atoms)) ./ length(atoms)
    probadist = FixedAtomsProbabilityDistribution(atoms, weights)
    return probadist
end

function compute_probability_distribution(
    perturbed::AbstractPerturbed, θ::AbstractArray{<:Real}; kwargs...
)
    Z_samples = sample_perturbations(perturbed, θ)
    return compute_probability_distribution(perturbed, θ, Z_samples; kwargs...)
end

function (perturbed::AbstractPerturbed)(θ::AbstractArray{<:Real}; kwargs...)
    probadist = compute_probability_distribution(perturbed, θ; kwargs...)
    return compute_expectation(probadist)
end

function ChainRulesCore.rrule(
    ::typeof(compute_probability_distribution),
    perturbed::AbstractPerturbed,
    θ::AbstractArray{<:Real};
    kwargs...,
)
    return error("Not implemented")
end
