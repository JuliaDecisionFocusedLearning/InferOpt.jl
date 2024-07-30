"""
    PerturbedMultiplicative{P,G,O,R,S,parallel} <: AbstractPerturbed{parallel}

Differentiable multiplicative perturbation of a black-box oracle:
the input undergoes `θ -> θ ⊙ exp[εZ - ε²/2]` where `Z ∼ perturbation`.

This [`AbstractOptimizationLayer`](@ref) is compatible with [`FenchelYoungLoss`](@ref),
if the oracle is an optimization maximizer with a linear objective.

Reference: <https://arxiv.org/abs/2207.13513>

See [`AbstractPerturbed`](@ref) for more details.

# Specific field
- `ε:Float64`: size of the perturbation
"""
struct PerturbedMultiplicative{P,G,O,R<:AbstractRNG,S<:Union{Nothing,Int},parallel} <:
       AbstractPerturbed{O,parallel}
    perturbation::P
    grad_logdensity::G
    oracle::O
    rng::R
    seed::S
    nb_samples::Int
    ε::Float64
end

function Base.show(io::IO, perturbed::PerturbedMultiplicative)
    (; oracle, ε, rng, seed, nb_samples, perturbation) = perturbed
    perturb = isnothing(perturbation) ? "Normal(0, 1)" : "$perturbation"
    return print(
        io,
        "PerturbedMultiplicative($oracle, $ε, $nb_samples, $(typeof(rng)), $seed, $perturb)",
    )
end

"""
    PerturbedMultiplicative(oracle[; ε, nb_samples, seed, is_parallel, perturbation, grad_logdensity, rng])

[`PerturbedMultiplicative`](@ref) constructor.

# Arguments
- `oracle`: the black-box oracle we want to differentiate through.
    It should be a linear maximizer if you want to use it inside a [`FenchelYoungLoss`].

# Keyword arguments (optional)
- `ε=1.0`: size of the perturbation.
- `nb_samples::Int=1`: number of perturbation samples drawn at each forward pass.
- `perturbation=nothing`: nothing by default. If you want to use a different distribution than a
    `Normal` for the perturbation `z`, give it here as a distribution-like object implementing
    the `rand` method. It should also implement `logdensityof` if `grad_logdensity` is not given.
- `grad_logdensity=nothing`: gradient function of `perturbation` w.r.t. `θ`.
    If set to nothing (default), it's computed using automatic differentiation.
- `seed::Union{Nothing,Int}=nothing`: seed of the perturbation.
    It is reset each time the forward pass is called,
    making it deterministic by always drawing the same perturbations.
    If you do not want this behaviour, set this field to `nothing`.
- `rng::AbstractRNG`=MersenneTwister(0): random number generator using the `seed`.
"""
function PerturbedMultiplicative(
    oracle::O;
    ε=1.0,
    nb_samples=1,
    seed::S=nothing,
    is_parallel=false,
    perturbation::P=nothing,
    grad_logdensity::G=nothing,
    rng::R=MersenneTwister(0),
) where {P,G,O,R<:AbstractRNG,S<:Union{Int,Nothing}}
    return PerturbedMultiplicative{P,G,O,R,S,is_parallel}(
        perturbation, grad_logdensity, oracle, rng, seed, nb_samples, float(ε)
    )
end

function sample_perturbations(perturbed::PerturbedMultiplicative, θ::AbstractArray)
    (; rng, seed, nb_samples, perturbation) = perturbed
    seed!(rng, seed)
    return [rand(rng, perturbation, size(θ)) for _ in 1:nb_samples]
end

function sample_perturbations(perturbed::PerturbedMultiplicative{Nothing}, θ::AbstractArray)
    (; rng, seed, nb_samples) = perturbed
    seed!(rng, seed)
    return [randn(rng, size(θ)) for _ in 1:nb_samples]
end

function compute_probability_distribution_from_samples(
    perturbed::PerturbedMultiplicative,
    θ::AbstractArray,
    Z_samples::Vector{<:AbstractArray};
    kwargs...,
)
    (; ε) = perturbed
    η_samples = [θ .* exp.(ε .* Z .- ε^2 / 2) for Z in Z_samples]
    atoms = compute_atoms(perturbed, η_samples; kwargs...)
    weights = ones(length(atoms)) ./ length(atoms)
    probadist = FixedAtomsProbabilityDistribution(atoms, weights)
    return probadist
end

function perturbation_grad_logdensity(
    ::RuleConfig,
    perturbed::PerturbedMultiplicative{Nothing,Nothing},
    θ::AbstractArray,
    Z::AbstractArray,
)
    (; ε) = perturbed
    return inv.(ε .* θ) .* Z
end

function _perturbation_logdensity(
    perturbed::PerturbedMultiplicative, θ::AbstractArray, η::AbstractArray
)
    (; ε, perturbation) = perturbed
    Z = (log.(η) .- log.(θ)) ./ ε .+ ε / 2
    return sum(logdensityof(perturbation, z) for z in Z)
end

function perturbation_grad_logdensity(
    rc::RuleConfig,
    perturbed::PerturbedMultiplicative{P,Nothing},
    θ::AbstractArray,
    Z::AbstractArray,
) where {P}
    (; ε) = perturbed
    η = θ .* exp.(ε .* Z .- ε^2 / 2)
    l, logdensity_pullback = rrule_via_ad(rc, _perturbation_logdensity, perturbed, θ, η)
    δperturbation_logdensity, δperturbed, δθ, δη = logdensity_pullback(one(l))
    return δθ
end
