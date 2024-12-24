"""
$TYPEDSIGNATURES

Data structure modeling the exponential of a continuous univariate random variable.
"""
struct ExponentialOf{D<:ContinuousUnivariateDistribution} <:
       ContinuousUnivariateDistribution
    dist::D
end

"""
$TYPEDSIGNATURES
"""
function Random.rand(rng::AbstractRNG, d::ExponentialOf)
    return exp(rand(rng, d.dist))
end

"""
$TYPEDSIGNATURES

Return the log-density of the [`ExponentialOf`](@ref) distribution at `x`.
It is equal to ``logpdf(d, log(x)) - log(x)``
"""
function Distributions.logpdf(d::ExponentialOf, x::Real)
    return logpdf(d.dist, log(x)) - log(x)
end
