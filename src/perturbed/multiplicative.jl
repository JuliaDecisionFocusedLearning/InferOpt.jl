"""
    PerturbedMultiplicative{F}

Differentiable log-normal perturbation of a black-box optimizer of type `F`: the input undergoes `θ -> θ ⊙ exp[εZ - ε²/2]` where `Z ∼ N(0, I)`.

See also: [`AbstractPerturbed`](@ref).

Reference: preprint coming soon.
"""
struct PerturbedMultiplicative{F,R<:AbstractRNG,S<:Union{Nothing,Int}} <: AbstractPerturbed{F}
    maximizer::F
    ε::Float64
    nb_samples::Int
    rng::R
    seed::S
end

function Base.show(io::IO, perturbed::PerturbedMultiplicative)
    (; maximizer, ε, rng, seed, nb_samples) = perturbed
    return print(
        io, "PerturbedMultiplicative($maximizer, $ε, $nb_samples, $(typeof(rng)), $seed)"
    )
end

"""
    PerturbedMultiplicative(maximizer[; ε=1.0, nb_samples=1])

Shorter constructor with defaults.
"""
function PerturbedMultiplicative(
    maximizer; ε=1.0, epsilon=nothing, nb_samples=1, rng=MersenneTwister(0), seed=nothing
)
    if isnothing(epsilon)
        return PerturbedMultiplicative(maximizer, float(ε), nb_samples, rng, seed)
    else
        return PerturbedMultiplicative(maximizer, float(epsilon), nb_samples, rng, seed)
    end
end

## Forward pass

function perturb_and_optimize(
    perturbed::PerturbedMultiplicative,
    θ::AbstractArray{<:Real},
    Z::AbstractArray{<:Real};
    kwargs...,
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
    θ::AbstractArray{<:Real};
    kwargs...,
)
    (; ε) = perturbed
    Z_samples = sample_perturbations(perturbed, θ)
    probadist = compute_probability_distribution(perturbed, θ, Z_samples; kwargs...)
    function perturbed_multiplicative_probadist_pullback(probadist_tangent)
        weigths_tangent = probadist_tangent.weights
        dθ = inv.(ε .* θ) .* sum(wt * Z for (wt, Z) in zip(weigths_tangent, Z_samples))
        return NoTangent(), NoTangent(), dθ
    end
    return probadist, perturbed_multiplicative_probadist_pullback
end
