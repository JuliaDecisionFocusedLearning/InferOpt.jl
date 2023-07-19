"""
    PerturbedMultiplicative <: AbstractPerturbed

Differentiable log-normal perturbation of a black-box oracle:
the input undergoes `θ -> θ ⊙ exp[εZ - ε²/2]` where `Z ∼ perturbation`.

Reference: <https://arxiv.org/abs/2207.13513>

See [`AbstractPerturbed`](@ref) for more details.

# Fields
- `oracle`
- `ε`
- `nb_samples`
- `rng`
- `seed`
- `perturbation`
"""
struct PerturbedMultiplicative{P,G,O,R<:AbstractRNG,S<:Union{Nothing,Int},parallel} <:
       AbstractPerturbed{parallel}
    oracle::O
    ε::Float64
    nb_samples::Int
    rng::R
    seed::S
    perturbation::P
    grad_logdensity::G
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
    PerturbedMultiplicative(maximizer[; ε=1.0, nb_samples=1])
"""
function PerturbedMultiplicative(
    oracle::F;
    ε=1.0,
    nb_samples=1,
    rng::R=MersenneTwister(0),
    seed::S=nothing,
    is_parallel=false,
    perturbation::P=nothing,
    grad_logdensity::G=nothing,
) where {F,R,S,P,G}
    return PerturbedMultiplicative{P,G,F,R,S,is_parallel}(
        oracle, float(ε), nb_samples, rng, seed, perturbation, grad_logdensity
    )
end

function sample_perturbations(perturbed::PerturbedMultiplicative, θ::AbstractArray)
    (; rng, seed, nb_samples, perturbation) = perturbed
    seed!(rng, seed)
    return [rand(rng, perturbation(θ)) for _ in 1:nb_samples]
end

function sample_perturbations(perturbed::PerturbedMultiplicative{Nothing}, θ::AbstractArray)
    (; rng, seed, nb_samples) = perturbed
    seed!(rng, seed)
    return [randn(rng, size(θ)) for _ in 1:nb_samples]
end

function compute_probability_distribution_from_samples(
    perturbed::PerturbedMultiplicative, θ, Z_samples::Vector{<:AbstractArray}; kwargs...
)
    (; ε) = perturbed
    # Z_samples = sample_perturbations(perturbed, θ)
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
    return logdensityof(perturbation(θ), Z)
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
