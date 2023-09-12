"""
    AbstractPerturbed{F,parallel} <: AbstractOptimizationLayer

Differentiable perturbation of a black box optimizer of type `F`.

The parameter `parallel` is a boolean value indicating if the perturbations are run in parallel.
This is particularly useful if your black box optimizer running time is high.

# Available implementations:
- [`PerturbedAdditive`](@ref)
- [`PerturbedMultiplicative`](@ref)
- [`PerturbedOracle`](@ref)

# These three subtypes share the following fields:
- `oracle`: black box (optimizer)
- `perturbation::P`: perturbation distribution of the input θ
- `grad_logdensity::G`: gradient of the log density `perturbation` w.r.t. input θ
- `nb_samples::Int`: number of perturbation samples drawn at each forward pass
- `seed::Union{Nothing,Int}`: seed of the perturbation.
    It is reset each time the forward pass is called,
    making it deterministic by always drawing the same perturbations.
    If you do not want this behaviour, set this field to `nothing`.
- `rng::AbstractRNG`: random number generator using the `seed`.

!!! warning
    The `perturbation` field does not mean the same thing for a [`PerturbedOracle`](@ref)
    than for a [`PerturbedAdditive`](@ref)/[`PerturbedMultiplicative`](@ref). See their respective docs.
"""
abstract type AbstractPerturbed{O,parallel} <: AbstractOptimizationLayer end

# Non parallelized version
function compute_atoms(
    perturbed::AbstractPerturbed{O,false}, η_samples::Vector{<:AbstractArray}; kwargs...
) where {O}
    return [perturbed.oracle(η; kwargs...) for η in η_samples]
end

# Parallelized version
function compute_atoms(
    perturbed::AbstractPerturbed{O,true}, η_samples::Vector{<:AbstractArray}; kwargs...
) where {O}
    return ThreadsX.map(η -> perturbed.oracle(η; kwargs...), η_samples)
end

"""
    sample_perturbations(perturbed::AbstractPerturbed, θ::AbstractArray)

Draw `nb_samples` random perturbations from the `perturbation` distribution.
"""
function sample_perturbations end

"""
    perturbation_grad_logdensity(
        ::RuleConfig,
        ::AbstractPerturbed,
        θ::AbstractArray,
        sample::AbstractArray,
    )

Compute de gradient w.r.t to the input `θ` of the logdensity of the perturbed input
distribution evaluated in the observed perturbation sample `η`.
"""
function perturbation_grad_logdensity end

"""
    compute_probability_distribution_from_samples(
        ::AbstractPerturbed,
        θ::AbstractArray,
        samples::Vector{<:AbstractArray};
        kwargs...,
    )

Create a probability distributions from `samples` drawn from `perturbation`.
"""
function compute_probability_distribution_from_samples end

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

Forward pass. Compute the expectation of the underlying distribution.
"""
function (perturbed::AbstractPerturbed)(
    θ::AbstractArray; autodiff_variance_reduction::Bool=true, kwargs...
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
    autodiff_variance_reduction::Bool=true,
    kwargs...,
)
    η_samples = sample_perturbations(perturbed, θ)
    y_dist = compute_probability_distribution_from_samples(
        perturbed, θ, η_samples; kwargs...
    )

    ∇logp_samples = [perturbation_grad_logdensity(rc, perturbed, θ, η) for η in η_samples]

    M = perturbed.nb_samples
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
