"""
$TYPEDEF

Abstract type for a perturbation.
It's a function that takes a parameter `θ` and returns a perturbed parameter by a distribution `perturbation_dist`.

!!! warning
    All subtypes should implement a `perturbation_dist` field, which is a `ContinuousUnivariateDistribution`.

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
$TYPEDEF

Method with parameters to compute the gradient of the logdensity of η = θ + εZ w.r.t. θ., with Z ∼ N(0, 1).

# Fields
$TYPEDFIELDS
"""
struct NormalAdditiveGradLogdensity
    "perturbation size"
    ε::Float64
end

function NormalAdditiveGradLogdensity(pdc::AdditivePerturbation)
    return NormalAdditiveGradLogdensity(pdc.ε)
end

"""
$TYPEDSIGNATURES

Compute the gradient of the logdensity of η = θ + εZ w.r.t. θ., with Z ∼ N(0, 1).
"""
function (f::NormalAdditiveGradLogdensity)(η::AbstractArray, θ::AbstractArray)
    (; ε) = f
    return ((η .- θ) ./ ε^2,)
end

"""
$TYPEDEF

Multiplicative perturbation: θ ↦ θ ⊙ exp(εZ - shift)

# Fields
$TYPEDFIELDS
"""
struct MultiplicativePerturbation{F}
    "base distribution for the perturbation"
    perturbation_dist::F
    "perturbation size"
    ε::Float64
    "optional shift to have 0 mean, default value is ε²/2"
    shift::Float64
end

"""
$TYPEDSIGNATURES

Constructor for [`MultiplicativePerturbation`](@ref).
"""
function MultiplicativePerturbation(perturbation_dist, ε, shift=ε^2 / 2)
    return MultiplicativePerturbation(perturbation_dist, ε, shift)
end

"""
$TYPEDSIGNATURES

Apply the multiplicative perturbation to the parameter `θ`.
"""
function (pdc::MultiplicativePerturbation)(θ::AbstractArray)
    (; perturbation_dist, ε, shift) = pdc
    return product_distribution(θ .* ExponentialOf(ε * perturbation_dist - shift))
end

"""
$TYPEDEF

Method with parameters to compute the gradient of the logdensity of η = θ ⊙ exp(εZ - shift) w.r.t. θ., with Z ∼ N(0, 1).

# Fields
$TYPEDFIELDS
"""
struct NormalMultiplicativeGradLogdensity
    "perturbation size"
    ε::Float64
    "optional shift to have 0 mean"
    shift::Float64
end

function NormalMultiplicativeGradLogdensity(pdc::MultiplicativePerturbation)
    return NormalMultiplicativeGradLogdensity(pdc.ε, pdc.shift)
end

function NormalMultiplicativeGradLogdensity(ε::Float64, shift=ε^2 / 2)
    return NormalMultiplicativeGradLogdensity(ε, shift)
end

"""
$TYPEDSIGNATURES

Compute the gradient of the logdensity of η = θ ⊙ exp(εZ - shift) w.r.t. θ., with Z ∼ N(0, 1).

!!! warning
    η should be a realization of θ, i.e. should be of the same sign.
"""
function (f::NormalMultiplicativeGradLogdensity)(η::AbstractArray, θ::AbstractArray)
    (; ε, shift) = f
    return (inv.(ε^2 .* θ) .* (log.(abs.(η)) - log.(abs.(θ)) .+ shift),)
end
