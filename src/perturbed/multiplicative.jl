"""
    PerturbedMultiplicative{F}

Differentiable log-normal perturbation of a black-box optimizer: the input undergoes `θ -> θ ⊙ exp[εZ - ε²/2]` where `Z ∼ N(0, I)`.

See also: [`AbstractPerturbed{F}`](@ref).
"""
struct PerturbedMultiplicative{F,R<:AbstractRNG,S<:Union{Nothing,Int}} <: AbstractPerturbed{F}
    maximizer::F
    ε::Float64
    rng::R
    seed::S
    nb_samples::Int
end

function Base.show(io::IO, perturbed::PerturbedMultiplicative)
    (; maximizer, ε, rng, seed, nb_samples) = perturbed
    print(io, "PerturbedMultiplicative($maximizer, $ε, $(typeof(rng)), $seed, $nb_samples)")
end

function PerturbedMultiplicative(
    maximizer; ε=1.0, rng=MersenneTwister(0), seed=nothing, nb_samples=2
)
    return PerturbedMultiplicative(maximizer, float(ε), rng, seed, nb_samples)
end

## Forward pass

function (perturbed::PerturbedMultiplicative)(θ::AbstractArray, Z::AbstractArray; kwargs...)
    (; maximizer, ε) = perturbed
    θ_perturbed = θ .* exp.(ε .* Z .- ε^2)
    y = maximizer(θ_perturbed; kwargs...)
    return y
end

## Fenchel-Young loss

function compute_y_and_F(
    perturbed::PerturbedMultiplicative, θ::AbstractArray, Z::AbstractArray; kwargs...
)
    (; maximizer, ε) = perturbed
    eZ = exp.(ε .* Z .- ε^2)
    θ_perturbed = θ .* eZ
    y = maximizer(θ_perturbed; kwargs...)
    F = dot(θ_perturbed, y)
    return y .* eZ, F
end

## Backward pass

function ChainRulesCore.rrule(
    perturbed_composition::PerturbedComposition{F,P,G}, θ::AbstractArray{<:Real}; kwargs...
) where {F,P<:PerturbedMultiplicative{F},G}
    (; perturbed, g) = perturbed_composition
    (; maximizer, ε) = perturbed
    Z_samples = sample_perturbations(perturbed, θ)
    y_samples = [maximizer(θ .* exp.(ε .* Z .- ε^2); kwargs...) for Z in Z_samples]
    gy_samples = [g(y; kwargs...) for y in y_samples]
    function perturbed_lognormal_composition_pullback(dgy)
        vjp =
            inv.(ε .* θ) .* mean(dot(dgy, gy) * Z for (Z, gy) in zip(Z_samples, gy_samples))
        return NoTangent(), vjp
    end
    return mean(gy_samples), perturbed_lognormal_composition_pullback
end

function ChainRulesCore.rrule(
    perturbed::PerturbedMultiplicative, θ::AbstractArray{<:Real}; kwargs...
)
    return rrule(PerturbedComposition(perturbed, identity), θ; kwargs...)
end
