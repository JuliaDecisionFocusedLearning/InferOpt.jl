"""
    AbstractPerturbed{F}

Differentiable perturbation of a black-box optimizer.

Subtypes:

- [`PerturbedNormal{F}`](@ref)
- [`PerturbedLogNormal{F}`](@ref)
"""
abstract type AbstractPerturbed{F} end

## Forward pass

function (perturbed::AbstractPerturbed)(θ::AbstractArray; kwargs...)
    (; M) = perturbed
    Z_samples = [randn(size(θ)) for _ in 1:M]
    return mean(perturbed(θ, Z; kwargs...) for Z in Z_samples)
end

## Fenchel-Young loss

function compute_y_and_F(perturbed::AbstractPerturbed, θ::AbstractArray; kwargs...)
    (; M) = perturbed
    Z_samples = [randn(size(θ)) for _ in 1:M]
    y_samples, F_samples = unzip(compute_y_and_F(perturbed, θ, Z) for Z in Z_samples)
    return mean(y_samples), mean(F_samples)
end
