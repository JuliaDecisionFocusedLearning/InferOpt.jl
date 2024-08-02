abstract type AbstractPerturbation <: ContinuousUnivariateDistribution end

function Random.rand(rng::AbstractRNG, perturbation::AbstractPerturbation)
    return rand(rng, perturbation.perturbation_dist)
end

struct AdditivePerturbation{F}
    perturbation_dist::F
    ε::Float64
end

"""
θ + εZ
"""
function (pdc::AdditivePerturbation)(θ::AbstractArray)
    (; perturbation_dist, ε) = pdc
    return product_distribution(θ .+ ε * perturbation_dist)
end

"""
θ ⊙ exp(εZ - ε²/2)
"""
struct MultiplicativePerturbation{F}
    perturbation_dist::F
    ε::Float64
end

function (pdc::MultiplicativePerturbation)(θ::AbstractArray)
    (; perturbation_dist, ε) = pdc
    return product_distribution(θ .* ExponentialOf(ε * perturbation_dist - ε^2 / 2))
end
