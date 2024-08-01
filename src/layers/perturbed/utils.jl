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
