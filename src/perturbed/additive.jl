"""
    PerturbedAdditive{F}

Differentiable normal perturbation of a black-box optimizer of type `F`: the input undergoes `θ -> θ + εZ` where `Z ∼ N(0, I)`.

See also: [`AbstractPerturbed`](@ref).

Reference: <https://arxiv.org/abs/2002.08676>
"""
struct PerturbedAdditive{F,R<:AbstractRNG,S<:Union{Nothing,Int},parallel} <:
       AbstractPerturbed{parallel}
    maximizer::F
    ε::Float64
    nb_samples::Int
    rng::R
    seed::S

    function PerturbedAdditive{F,R,S,parallel}(
        maximizer::F,
        ε::Float64,
        nb_samples::Int,
        rng::R,
        seed::S,
    ) where {F,R<:AbstractRNG,S<:Union{Nothing,Int},parallel}
        @assert parallel isa Bool
        return new{F,R,S,parallel}(maximizer, ε, nb_samples, rng, seed)
    end
end

function Base.show(io::IO, perturbed::PerturbedAdditive)
    (; maximizer, ε, rng, seed, nb_samples) = perturbed
    return print(
        io,
        "PerturbedAdditive($maximizer, $ε, $nb_samples, $(typeof(rng)), $seed)",
    )
end

"""
    PerturbedAdditive(maximizer[; ε=1.0, nb_samples=1])

Shorter constructor with defaults.
"""
function PerturbedAdditive(
    maximizer::F;
    ε=1.0,
    nb_samples=1,
    rng::R=MersenneTwister(0),
    seed::S=nothing,
    is_parallel=false,
) where {F,R,S}
    return PerturbedAdditive{F,R,S,is_parallel}(maximizer, float(ε), nb_samples, rng, seed)
end

## Forward pass

function perturb_and_optimize(
    perturbed::PerturbedAdditive,
    θ::AbstractArray{<:Real},
    Z::AbstractArray{<:Real};
    kwargs...,
)
    (; maximizer, ε) = perturbed
    θ_perturbed = θ .+ ε .* Z
    y = maximizer(θ_perturbed; kwargs...)
    return y
end

## Backward pass

function ChainRulesCore.rrule(
    ::typeof(compute_probability_distribution),
    perturbed::PerturbedAdditive,
    θ::AbstractArray{<:Real};
    kwargs...,
)
    (; ε) = perturbed
    Z_samples = sample_perturbations(perturbed, θ)
    probadist = compute_probability_distribution(perturbed, θ, Z_samples; kwargs...)
    function perturbed_additive_probadist_pullback(probadist_tangent)
        weigths_tangent = probadist_tangent.weights
        dθ = inv(ε) * mean(wt * Z for (wt, Z) in zip(weigths_tangent, Z_samples))
        return NoTangent(), NoTangent(), dθ
    end
    return probadist, perturbed_additive_probadist_pullback
end
