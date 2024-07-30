struct ExponentialOf{D<:ContinuousUnivariateDistribution} <:
       ContinuousUnivariateDistribution
    dist::D
end

function Random.rand(rng::AbstractRNG, d::ExponentialOf)
    return exp(rand(rng, d.dist))
end

function Distributions.logpdf(d::ExponentialOf, x::Real)
    return logpdf(d.dist, log(x)) - log(x)
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
