"""
$TYPEDEF

Differentiable perturbed black-box oracle. The `oracle` input `θ` is perturbed as `η ∼ perturbation(⋅|θ)`.
[`PerturbedAdditive`](@ref) is a special case of `PerturbedOracle` with `perturbation(θ) = MvNormal(θ, ε * I)`.
[`PerturbedMultiplicative`] is also a special case of `PerturbedOracle`.

See [`AbstractPerturbed`](@ref) for more details about its fields.

# Fields
$TYPEDFIELDS
"""
struct PerturbedOracle{P,G,O,R<:AbstractRNG,S<:Union{Nothing,Int},parallel} <:
       AbstractPerturbed{O,parallel}
    perturbation::P
    grad_logdensity::G
    oracle::O
    rng::R
    seed::S
    nb_samples::Int
end

"""
    PerturbedOracle(perturbation, oracle[; grad_logdensity, rng, seed, is_parallel, nb_samples])

[`PerturbedOracle`](@ref) constructor.

# Arguments
- `oracle`: the black-box oracle we want to differentiate through
- `perturbation`: should be a callable such that `perturbation(θ)` is a distribution-like
    object that can be sampled with `rand`.
    It should also implement `logdensityof` if `grad_logdensity` is not given.

# Keyword arguments (optional)
- `grad_logdensity=nothing`: gradient function of `perturbation` w.r.t. `θ`.
    If set to nothing (default), it's computed using automatic differentiation.
- `nb_samples::Int=1`: number of perturbation samples drawn at each forward pass
- `seed::Union{Nothing,Int}=nothing`: seed of the perturbation.
    It is reset each time the forward pass is called,
    making it deterministic by always drawing the same perturbations.
    If you do not want this behaviour, set this field to `nothing`.
- `rng::AbstractRNG`=MersenneTwister(0): random number generator using the `seed`.

!!! info
    If you have access to the analytical expression of `grad_logdensity` it is recommended to
    give it, as it will be computationally faster.
"""
function PerturbedOracle(
    oracle::O,
    perturbation::P;
    grad_logdensity::G=nothing,
    nb_samples::Int=1,
    seed::S=nothing,
    is_parallel::Bool=false,
    rng::R=MersenneTwister(0),
) where {P,G,O,R<:AbstractRNG,S<:Union{Int,Nothing}}
    return PerturbedOracle{P,G,O,R,S,is_parallel}(
        perturbation, grad_logdensity, oracle, rng, seed, nb_samples
    )
end

function Base.show(io::IO, po::PerturbedOracle)
    (; oracle, perturbation, rng, seed, nb_samples) = po
    return print(
        io, "PerturbedOracle($perturbation, $oracle, $nb_samples, $(typeof(rng)), $seed)"
    )
end

function sample_perturbations(po::PerturbedOracle, θ::AbstractArray)
    (; rng, seed, perturbation, nb_samples) = po
    seed!(rng, seed)
    η_samples = [rand(rng, perturbation(θ)) for _ in 1:nb_samples]
    return η_samples
end

function compute_probability_distribution_from_samples(
    perturbed::PerturbedOracle, θ, η_samples::Vector{<:AbstractArray}; kwargs...
)
    atoms = compute_atoms(perturbed, η_samples; kwargs...)
    weights = ones(length(atoms)) ./ length(atoms)
    probadist = FixedAtomsProbabilityDistribution(atoms, weights)
    return probadist
end

function _perturbation_logdensity(po::PerturbedOracle, θ::AbstractArray, η::AbstractArray)
    return logdensityof(po.perturbation(θ), η)
end

function perturbation_grad_logdensity(
    rc::RuleConfig, po::PerturbedOracle{P,Nothing}, θ::AbstractArray, η::AbstractArray
) where {P}
    l, logdensity_pullback = rrule_via_ad(rc, _perturbation_logdensity, po, θ, η)
    δperturbation_logdensity, δpo, δθ, δη = logdensity_pullback(one(l))
    return δθ
end
