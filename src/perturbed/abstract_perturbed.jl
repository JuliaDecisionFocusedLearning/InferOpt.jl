"""
    AbstractPerturbed{parallel,P,G} <: AbstractOptimizationLayer

Differentiable perturbation of a black box optimizer.

The parameter `parallel` is a boolean value, equal to true if the perturbations are run in parallel.

# Available implementations

- [`PerturbedAdditive`](@ref)
- [`PerturbedMultiplicative`](@ref)
- [`PerturbedOracle`](@ref)

These three subtypes share the following fields:

- `oracle`: black box (optimizer)
- `perturbation::P` -> doesn't mean the same thing depending on the implementation, use different names ?
- `grad_logdensity::G`
- `nb_samples::Int`: number of random samples for Monte-Carlo computations
- `rng::AbstractRNG`: random number generator
- `seed::Union{Nothing,Int}`: random seed
"""
abstract type AbstractPerturbed{parallel} <: AbstractOptimizationLayer end

"""
    sample_perturbations(perturbed::AbstractPerturbed, θ::AbstractArray)

Draw `nb_samples` random perturbations from perturbation(θ).
"""
function sample_perturbations end

"""
perturbation_grad_logdensity::RuleConfig,
    perturbed::AbstractPerturbed,
    θ::AbstractArray,
    η::AbstractArray,
)
"""
function perturbation_grad_logdensity end

# TODO: remove this, all imlementations have the nb_samples field
function get_nb_samples(perturbed::AbstractPerturbed)
    return perturbed.nb_samples
end

function compute_atoms(
    perturbed::AbstractPerturbed{false}, η_samples::Vector{<:AbstractArray}; kwargs...
)
    return [perturbed.oracle(η; kwargs...) for η in η_samples]
end

function compute_atoms(
    perturbed::AbstractPerturbed{true}, η_samples::Vector{<:AbstractArray}; kwargs...
)
    return ThreadsX.map(η -> perturbed.oracle(η; kwargs...), η_samples)
end

function compute_probability_distribution_from_samples(
    perturbed::AbstractPerturbed, θ, η_samples::Vector{<:AbstractArray}; kwargs...
)
    atoms = compute_atoms(perturbed, η_samples; kwargs...)
    weights = ones(length(atoms)) ./ length(atoms)
    probadist = FixedAtomsProbabilityDistribution(atoms, weights)
    return probadist
end

"""
    compute_probability_distribution(perturbed::AbstractPerturbed, θ; kwargs...)

Turn random perturbations of `θ` into a distribution on polytope vertices.

Keyword arguments are passed to the underlying linear maximizer.
"""
function compute_probability_distribution(
    perturbed::AbstractPerturbed,
    θ::AbstractArray;
    autodiff_variance_reduction::Bool=false,
    kwargs...,
)
    η_samples = sample_perturbations(perturbed, θ)
    return compute_probability_distribution_from_samples(perturbed, θ, η_samples; kwargs...)
end

# Forward pass

"""
    (perturbed::AbstractPerturbed)(θ; kwargs...)

Apply `compute_probability_distribution(perturbed, θ; kwargs...)` and return the expectation.
"""
function (perturbed::AbstractPerturbed)(
    θ::AbstractArray; autodiff_variance_reduction::Bool=false, kwargs...
)
    probadist = compute_probability_distribution(
        perturbed, θ; autodiff_variance_reduction, kwargs...
    )
    return compute_expectation(probadist)
end

function perturbation_grad_logdensity(
    ::RuleConfig, perturbed::AbstractPerturbed, θ::AbstractArray, η::AbstractArray
)
    return perturbed.grad_logdensity(θ, η)
end

# Backward pass

function ChainRulesCore.rrule(
    rc::RuleConfig,
    ::typeof(compute_probability_distribution),
    perturbed::AbstractPerturbed,
    θ::AbstractArray;
    autodiff_variance_reduction::Bool=false,
    kwargs...,
)
    η_samples = sample_perturbations(perturbed, θ)
    y_dist = compute_probability_distribution_from_samples(
        perturbed, θ, η_samples; kwargs...
    )

    ∇logp_samples = [perturbation_grad_logdensity(rc, perturbed, θ, η) for η in η_samples]

    M = get_nb_samples(perturbed)
    function perturbed_oracle_dist_pullback(δy_dist)
        weights = y_dist.weights
        δy_weights = δy_dist.weights
        δy_sum = sum(δy_weights)
        δθ = sum(
            map(1:M) do i
                δyᵢ, ∇logpᵢ, w = δy_weights[i], ∇logp_samples[i], weights[i]
                if autodiff_variance_reduction
                    bᵢ = M == 1 ? 0 * δy_sum : (δy_sum - δyᵢ) / (M - 1)
                    return w * (δyᵢ - bᵢ) * ∇logpᵢ
                else
                    return w * δyᵢ * ∇logpᵢ
                end
            end,
        )
        return NoTangent(), NoTangent(), δθ
    end

    return y_dist, perturbed_oracle_dist_pullback
end
