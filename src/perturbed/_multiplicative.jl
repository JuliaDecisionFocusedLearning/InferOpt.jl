"""
    PerturbedMultiplicative <: AbstractPerturbed

Differentiable log-normal perturbation of a black-box maximizer: the input undergoes `θ -> θ ⊙ exp[εZ - ε²/2]` where `Z ∼ N(0, I)`.

Reference: <https://arxiv.org/abs/2207.13513>

See [`AbstractPerturbed`](@ref) for more details.
"""
struct PerturbedMultiplicative{F,R<:AbstractRNG,S<:Union{Nothing,Int},parallel} <:
       AbstractPerturbed{parallel}
    maximizer::F
    ε::Float64
    nb_samples::Int
    rng::R
    seed::S

    function PerturbedMultiplicative{F,R,S,parallel}(
        maximizer::F, ε::Float64, nb_samples::Int, rng::R, seed::S
    ) where {F,R<:AbstractRNG,S<:Union{Nothing,Int},parallel}
        @assert parallel isa Bool
        return new{F,R,S,parallel}(maximizer, ε, nb_samples, rng, seed)
    end
end

function Base.show(io::IO, perturbed::PerturbedMultiplicative)
    (; maximizer, ε, rng, seed, nb_samples) = perturbed
    return print(
        io, "PerturbedMultiplicative($maximizer, $ε, $nb_samples, $(typeof(rng)), $seed)"
    )
end

"""
    PerturbedMultiplicative(maximizer[; ε=1.0, nb_samples=1])
"""
function PerturbedMultiplicative(
    maximizer::F;
    ε=1.0,
    nb_samples=1,
    rng::R=MersenneTwister(0),
    seed::S=nothing,
    is_parallel=false,
) where {F,R,S}
    return PerturbedMultiplicative{F,R,S,is_parallel}(
        maximizer, float(ε), nb_samples, rng, seed
    )
end

## Forward pass

function perturb_and_optimize(
    perturbed::PerturbedMultiplicative, θ::AbstractArray, Z::AbstractArray; kwargs...
)
    (; maximizer, ε) = perturbed
    θ_perturbed = θ .* exp.(ε .* Z .- ε^2)
    y = maximizer(θ_perturbed; kwargs...)
    return y
end

## Backward pass

function ChainRulesCore.rrule(
    ::typeof(compute_probability_distribution),
    perturbed::PerturbedMultiplicative,
    θ::AbstractArray;
    kwargs...,
)
    (; ε) = perturbed
    Z_samples = sample_perturbations(perturbed, θ)
    probadist = compute_probability_distribution(perturbed, θ, Z_samples; kwargs...)
    function perturbed_multiplicative_probadist_pullback(probadist_tangent)
        weights_tangent = probadist_tangent.weights
        if length(weights_tangent) != length(Z_samples)
            throw(ArgumentError("Probadist tangent has invalid number of atoms"))
        end
        dθ = inv.(ε .* θ) .* mean(wt * Z for (wt, Z) in zip(weights_tangent, Z_samples))
        return NoTangent(), NoTangent(), dθ
    end
    return probadist, perturbed_multiplicative_probadist_pullback
end
