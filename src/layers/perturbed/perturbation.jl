"""
$TYPEDEF

Abstract type for a perturbation.
It's a function that takes a parameter `θ` and returns a perturbed parameter by a distribution `perturbation_dist`.

All subtypes should have a `perturbation_dist`

# Existing implementations
- [`AdditivePerturbation`](@ref)
- [`MultiplicativePerturbation`](@ref)
"""
abstract type AbstractPerturbation <: ContinuousUnivariateDistribution end

"""
$TYPEDSIGNATURES
"""
function Random.rand(rng::AbstractRNG, perturbation::AbstractPerturbation)
    return rand(rng, perturbation.perturbation_dist)
end

"""
$TYPEDEF

Additive perturbation: θ ↦ θ + εZ, where Z is a random variable following `perturbation_dist`.

# Fields
$TYPEDFIELDS
"""
struct AdditivePerturbation{F}
    "base distribution for the perturbation"
    perturbation_dist::F
    "perturbation size"
    ε::Float64
end

"""
$TYPEDSIGNATURES

Apply the additive perturbation to the parameter `θ`.
"""
function (pdc::AdditivePerturbation)(θ::AbstractArray)
    (; perturbation_dist, ε) = pdc
    return product_distribution(θ .+ ε * perturbation_dist)
end

"""
$TYPEDSIGNATURES

Compute the gradient of the logdensity of η = θ + εZ w.r.t. θ., with Z ∼ N(0, 1).
"""
function normal_additive_grad_logdensity(ε, η, θ)
    return ((η .- θ) ./ ε^2,)
end

"""
$TYPEDEF

Multiplicative perturbation: θ ↦ θ ⊙ exp(εZ - ε²/2)

# Fields
$TYPEDFIELDS
"""
struct MultiplicativePerturbation{F}
    "base distribution for the perturbation"
    perturbation_dist::F
    "perturbation size"
    ε::Float64
end

"""
$TYPEDSIGNATURES

Apply the multiplicative perturbation to the parameter `θ`.
"""
function (pdc::MultiplicativePerturbation)(θ::AbstractArray)
    (; perturbation_dist, ε) = pdc
    return product_distribution(θ .* ExponentialOf(ε * perturbation_dist - ε^2 / 2))
end
"""
$TYPEDSIGNATURES

Compute the gradient of the logdensity of η = θ ⊙ exp(εZ - ε²/2) w.r.t. θ., with Z ∼ N(0, 1).
!!! warning
    η should be a relization of θ, i.e. should be of the same sign.
"""
function normal_multiplicative_grad_logdensity(ε, η, θ)
    return (inv.(ε^2 .* θ) .* (log.(abs.(η)) - log.(abs.(θ)) .+ (ε^2 / 2)),)
end
