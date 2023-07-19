"""
    PerturbedAdditive <: AbstractPerturbed

Differentiable normal perturbation of a black-box maximizer: the input undergoes `θ -> θ + εZ` where `Z ∼ N(0, I)`.

Reference: <https://arxiv.org/abs/2002.08676>

See [`AbstractPerturbed`](@ref) for more details.

# Fields
- `oracle`
- `ε`
- `nb_samples`
- `rng`
- `seed`
- `perturbation`
- `grad_logdensity`
"""
struct PerturbedAdditive{P,G,O,R<:AbstractRNG,S<:Union{Nothing,Int},parallel} <:
       AbstractPerturbed{parallel}
    oracle::O
    ε::Float64
    nb_samples::Int
    rng::R
    seed::S
    perturbation::P
    grad_logdensity::G

    # TODO: is this really necessary, the custom constructors enforces parallel to be a Bool. Probably not.
    function PerturbedAdditive{P,G,O,R,S,parallel}(
        oracle::O,
        ε::Float64,
        nb_samples::Int,
        rng::R,
        seed::S,
        perturbation::P,
        grad_logdensity::G,
    ) where {P,O,R<:AbstractRNG,S<:Union{Nothing,Int},parallel,G}
        @assert parallel isa Bool
        return new{P,G,O,R,S,parallel}(
            oracle, ε, nb_samples, rng, seed, perturbation, grad_logdensity
        )
    end
end

function Base.show(io::IO, perturbed::PerturbedAdditive)
    (; oracle, ε, rng, seed, nb_samples, perturbation) = perturbed
    perturb = isnothing(perturbation) ? "Normal(0, 1)" : "$perturbation"
    return print(
        io, "PerturbedAdditive($oracle, $ε, $nb_samples, $(typeof(rng)), $seed, $perturb)"
    )
end

"""
    PerturbedAdditive(maximizer[; ε=1.0, nb_samples=1])
"""
function PerturbedAdditive(
    oracle::O;
    ε=1.0,
    nb_samples=1,
    rng::R=MersenneTwister(0),
    seed::S=nothing,
    is_parallel::Bool=false,
    perturbation::P=nothing,
    grad_logdensity::G=nothing,
) where {O,R,S,P,G}
    return PerturbedAdditive{P,G,O,R,S,is_parallel}(
        oracle, float(ε), nb_samples, rng, seed, perturbation, grad_logdensity
    )
end

function sample_perturbations(perturbed::PerturbedAdditive, θ::AbstractArray)
    (; rng, seed, nb_samples, perturbation) = perturbed
    seed!(rng, seed)
    return [rand(rng, perturbation, size(θ)) for _ in 1:nb_samples]
end

function sample_perturbations(perturbed::PerturbedAdditive{Nothing}, θ::AbstractArray)
    (; rng, seed, nb_samples) = perturbed
    seed!(rng, seed)
    return [randn(rng, size(θ)) for _ in 1:nb_samples]
end

function compute_probability_distribution_from_samples(
    perturbed::PerturbedAdditive, θ, Z_samples::Vector{<:AbstractArray}; kwargs...
)
    (; ε) = perturbed
    η_samples = [θ .+ ε .* Z for Z in Z_samples]
    atoms = compute_atoms(perturbed, η_samples; kwargs...)
    weights = ones(length(atoms)) ./ length(atoms)
    probadist = FixedAtomsProbabilityDistribution(atoms, weights)
    return probadist
end

function perturbation_grad_logdensity(
    ::RuleConfig,
    perturbed::PerturbedAdditive{Nothing,Nothing},
    θ::AbstractArray,
    Z::AbstractArray,
)
    (; ε) = perturbed
    return Z ./ ε
end

function _perturbation_logdensity(
    perturbed::PerturbedAdditive, θ::AbstractArray, η::AbstractArray
)
    (; ε) = perturbed
    Z = (η .- θ) ./ ε
    return logdensityof(perturbed.perturbation, Z)
end

function perturbation_grad_logdensity(
    ::RuleConfig,
    perturbed::PerturbedAdditive{P,Nothing},
    θ::AbstractArray,
    Z::AbstractArray,
) where {P}
    (; ε) = perturbed
    η = θ .+ ε .* Z
    l, logdensity_pullback = rrule_via_ad(rc, _perturbation_logdensity, perturbed, θ, η)
    δperturbation_logdensity, δperturbed, δθ, δη = logdensity_pullback(one(l))
    return δθ
end
