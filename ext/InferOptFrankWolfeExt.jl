module InferOptFrankWolfeExt

using DifferentiableFrankWolfe: DiffFW, LinearMaximizationOracleWithKwargs
using InferOpt: InferOpt, RegularizedGeneric, FixedAtomsProbabilityDistribution
using InferOpt: compute_expectation, compute_probability_distribution
using LinearAlgebra: dot

## Forward pass

function InferOpt.compute_probability_distribution(
    dfw::DiffFW, θ::AbstractArray; frank_wolfe_kwargs=NamedTuple()
)
    weights, atoms = dfw.implicit(θ; frank_wolfe_kwargs=frank_wolfe_kwargs)
    probadist = FixedAtomsProbabilityDistribution(atoms, weights)
    return probadist
end

"""
    compute_probability_distribution(regularized::RegularizedGeneric, θ; kwargs...)

Construct a `DifferentiableFrankWolfe.DiffFW` struct and call `compute_probability_distribution` on it.

Keyword arguments are passed to the underlying linear maximizer.
"""
function InferOpt.compute_probability_distribution(
    regularized::RegularizedGeneric, θ::AbstractArray; kwargs...
)
    (; maximizer, Ω, Ω_grad, frank_wolfe_kwargs) = regularized
    f(y, θ) = Ω(y) - dot(θ, y)
    f_grad1(y, θ) = Ω_grad(y) - θ
    lmo = LinearMaximizationOracleWithKwargs(maximizer, kwargs)
    dfw = DiffFW(f, f_grad1, lmo)
    probadist = compute_probability_distribution(dfw, θ; frank_wolfe_kwargs)
    return probadist
end

"""
    (regularized::RegularizedGeneric)(θ; kwargs...)

Apply `compute_probability_distribution(regularized, θ)` and return the expectation.

Keyword arguments are passed to the underlying linear maximizer.
"""
function (regularized::RegularizedGeneric)(θ::AbstractArray; kwargs...)
    probadist = compute_probability_distribution(regularized, θ; kwargs...)
    return compute_expectation(probadist)
end

end
