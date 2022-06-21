"""
    PerturbedAdditive{F}

Differentiable normal perturbation of a black-box optimizer: the input undergoes `θ -> θ + εZ` where `Z ∼ N(0, I)`.

See also: [`AbstractPerturbed{F}`](@ref).
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
    print(io, "PerturbedAdditive($maximizer, $ε, $(typeof(rng)), $seed, $nb_samples)")
end

function PerturbedAdditive(
    maximizer; ε=1.0, rng=MersenneTwister(0), seed=nothing, nb_samples=2
)
    return PerturbedAdditive(maximizer, float(ε), rng, seed, nb_samples)
end

## Forward pass

function (perturbed::PerturbedAdditive)(θ::AbstractArray, Z::AbstractArray; kwargs...)
    (; maximizer, ε) = perturbed
    θ_perturbed = θ .+ ε .* Z
    y = maximizer(θ_perturbed; kwargs...)
    return y
end

## Fenchel-Young loss

function compute_y_and_F(
    perturbed::PerturbedAdditive, θ::AbstractArray, Z::AbstractArray; kwargs...
)
    (; maximizer, ε) = perturbed
    θ_perturbed = θ .+ ε .* Z
    y = maximizer(θ_perturbed; kwargs...)
    F = dot(θ_perturbed, y)
    return y, F
end

## Backward pass

function ChainRulesCore.rrule(
    perturbed_composition::PerturbedComposition{F,P,G}, θ::AbstractArray{<:Real}; kwargs...
) where {F,P<:PerturbedAdditive{F},G}
    (; perturbed, g) = perturbed_composition
    (; maximizer, ε) = perturbed
    Z_samples = sample_perturbations(perturbed, θ)
    y_samples = [maximizer(θ .+ ε .* Z; kwargs...) for Z in Z_samples]
    gy_samples = [g(y; kwargs...) for y in y_samples]
    function perturbed_normal_composition_pullback(dgy)
        vjp = inv(ε) * mean(dot(dgy, gy) * Z for (Z, gy) in zip(Z_samples, gy_samples))
        return NoTangent(), vjp
    end
    return mean(gy_samples), perturbed_normal_composition_pullback
end

function ChainRulesCore.rrule(
    perturbed::PerturbedAdditive, θ::AbstractArray{<:Real}; kwargs...
)
    return rrule(PerturbedComposition(perturbed, identity), θ; kwargs...)
end
