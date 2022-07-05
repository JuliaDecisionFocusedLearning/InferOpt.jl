"""
    PerturbedAdditive{F}

Differentiable normal perturbation of a black-box optimizer: the input undergoes `θ -> θ + εZ` where `Z ∼ N(0, I)`.

See also: [`AbstractPerturbed{F}`](@ref).

Reference: <https://arxiv.org/abs/2002.08676>
"""
struct PerturbedAdditive{F,R<:AbstractRNG,S<:Union{Nothing,Int}} <: AbstractPerturbed{F}
    maximizer::F
    ε::Float64
    rng::R
    seed::S
    nb_samples::Int
end

function Base.show(io::IO, perturbed::PerturbedAdditive)
    (; maximizer, ε, rng, seed, nb_samples) = perturbed
    return print(
        io, "PerturbedAdditive($maximizer, $ε, $(typeof(rng)), $seed, $nb_samples)"
    )
end

function PerturbedAdditive(
    maximizer; ε=1.0, epsilon=nothing, rng=MersenneTwister(0), seed=nothing, nb_samples=2
)
    if isnothing(epsilon)
        return PerturbedAdditive(maximizer, float(ε), rng, seed, nb_samples)
    else
        return PerturbedAdditive(maximizer, float(epsilon), rng, seed, nb_samples)
    end
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
        dθ = inv(ε) * sum(wt * Z for (wt, Z) in zip(weigths_tangent, Z_samples))
        return NoTangent(), NoTangent(), dθ
    end
    return probadist, perturbed_additive_probadist_pullback
end
