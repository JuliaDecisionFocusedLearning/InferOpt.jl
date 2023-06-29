"""
TODO
"""
struct PerturbedOracle{parallel,D,O,G,R<:AbstractRNG,S<:Union{Nothing,Int}}
    perturbation::D
    oracle::O
    grad_logdensity::G
    rng::R
    seed::S
    nb_samples::Int
end

function PerturbedOracle(
    perturbation::D,
    oracle::O;
    grad_logdensity::G=nothing,
    rng::R=MersenneTwister(0),
    seed::S=nothing,
    is_parallel::Bool=false,
    nb_samples::Int=1,
) where {D,O,G,R<:AbstractRNG,S<:Union{Int,Nothing}}
    return PerturbedOracle{is_parallel,D,O,G,R,S}(
        perturbation, oracle, grad_logdensity, rng, seed, nb_samples
    )
end

function Base.show(io::IO, po::PerturbedOracle)
    (; oracle, perturbation, rng, seed, nb_samples) = po
    return print(
        io, "PerturbedOracle($perturbation, $oracle, $nb_samples, $(typeof(rng)), $seed)"
    )
end

function perturbation_logdensity(po::PerturbedOracle, θ, η)
    return logdensityof(po.perturbation(θ), η)
end

function perturbation_grad_logdensity(
    rc::RuleConfig, po::PerturbedOracle{parallel,D,O,Nothing}, θ, η
) where {parallel,D,O}
    l, logdensity_pullback = rrule_via_ad(rc, perturbation_logdensity, po, θ, η)
    δperturbation_logdensity, δpo, δθ, δη = logdensity_pullback(one(l))
    return δθ
end

function perturbation_grad_logdensity(::RuleConfig, po::PerturbedOracle, θ, η)
    return po.grad_logdensity(θ, η)
end

function compute_atoms(po::PerturbedOracle{false}, η_samples; kwargs...)
    return [po.oracle(η; kwargs...) for η in η_samples]
end

function compute_atoms(po::PerturbedOracle{true}, η_samples; kwargs...)
    return ThreadsX.map(η -> po.oracle(η; kwargs...), η_samples)
end

## Forward pass

function sample_perturbations(po::PerturbedOracle, θ::AbstractArray)
    (; rng, seed, perturbation, nb_samples) = po
    seed!(rng, seed)
    η_samples = [rand(rng, perturbation(θ)) for _ in 1:nb_samples]
    return η_samples
end

function compute_probability_distribution(
    po::PerturbedOracle, θ; autodiff_variance_reduction::Bool=false, kwargs...
)
    η_samples = sample_perturbations(po, θ)
    y_samples = compute_atoms(po, η_samples; kwargs...)
    y_dist = FixedAtomsProbabilityDistribution(
        y_samples, ones(length(y_samples)) ./ length(y_samples)
    )
    return y_dist
end

function (po::PerturbedOracle)(θ; autodiff_variance_reduction::Bool=false, kwargs...)
    y_dist = compute_probability_distribution(po, θ; autodiff_variance_reduction, kwargs...)
    return compute_expectation(y_dist)
end

## Reverse pass

function ChainRulesCore.rrule(
    rc::RuleConfig,
    ::typeof(compute_probability_distribution),
    po::PerturbedOracle,
    θ;
    autodiff_variance_reduction::Bool=false,
    kwargs...,
)
    M = po.nb_samples
    η_samples = sample_perturbations(po, θ)
    y_samples = compute_atoms(po, η_samples; kwargs...)
    y_dist = FixedAtomsProbabilityDistribution(
        y_samples, ones(length(y_samples)) ./ length(y_samples)
    )
    ∇logp_samples = [perturbation_grad_logdensity(rc, po, θ, η) for η in η_samples]
    function perturbed_oracle_dist_pullback(δy_dist)
        δy_samples = δy_dist.weights
        δy_sum = sum(δy_samples)
        δθ_samples = map(1:M) do i
            δyᵢ, ∇logpᵢ = δy_samples[i], ∇logp_samples[i]
            bᵢ = M == 1 ? 0 * δy_sum : (δy_sum - δyᵢ) / (M - 1)
            return if autodiff_variance_reduction
                (δyᵢ - bᵢ) * ∇logpᵢ
            else
                δyᵢ * ∇logpᵢ
            end
        end
        return NoTangent(), NoTangent(), mean(δθ_samples)
    end
    return y_dist, perturbed_oracle_dist_pullback
end

function ChainRulesCore.rrule(
    rc::RuleConfig,
    po::PerturbedOracle,
    θ;
    autodiff_variance_reduction::Bool=false,
    kwargs...,
)
    M = po.nb_samples
    η_samples = sample_perturbations(po, θ)
    y_samples = compute_atoms(po, η_samples; kwargs...)
    y_dist = FixedAtomsProbabilityDistribution(
        y_samples, ones(length(y_samples)) ./ length(y_samples)
    )
    ∇logp_samples = [perturbation_grad_logdensity(rc, po, θ, η) for η in η_samples]
    y_sum = sum(y_samples)
    function perturbed_oracle_dist_pullback(δy)
        δθ_samples = map(1:M) do i
            yᵢ = y_samples[i]
            ∇logpᵢ = ∇logp_samples[i]
            bᵢ = M == 1 ? 0 * y_sum : (y_sum - yᵢ) / (M - 1)
            return if autodiff_variance_reduction
                (dot(δy, yᵢ) - dot(δy, bᵢ)) * ∇logpᵢ
            else
                dot(δy, yᵢ) * ∇logpᵢ
            end
        end
        return NoTangent(), mean(δθ_samples)
    end
    return compute_expectation(y_dist), perturbed_oracle_dist_pullback
end
