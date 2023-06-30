module InferOptFrankWolfeExt

using DifferentiableFrankWolfe: DiffFW, LinearMaximizationOracleWithKwargs
using InferOpt: InferOpt, RegularizedFrankWolfe, FixedAtomsProbabilityDistribution
using InferOpt: compute_expectation, compute_probability_distribution
using LinearAlgebra: dot

"""
    compute_probability_distribution(regularized::RegularizedFrankWolfe, θ; kwargs...)

Construct a `DifferentiableFrankWolfe.DiffFW` struct and call `compute_probability_distribution` on it.

Keyword arguments are passed to the underlying linear maximizer.
"""
function InferOpt.compute_probability_distribution(
    regularized::RegularizedFrankWolfe, θ::AbstractArray; kwargs...
)
    (; linear_maximizer, Ω, Ω_grad, frank_wolfe_kwargs) = regularized
    f(y, θ) = Ω(y) - dot(θ, y)
    f_grad1(y, θ) = Ω_grad(y) - θ
    lmo = LinearMaximizationOracleWithKwargs(linear_maximizer, kwargs)
    dfw = DiffFW(f, f_grad1, lmo)
    weights, atoms = dfw.implicit(θ; frank_wolfe_kwargs=frank_wolfe_kwargs)
    probadist = FixedAtomsProbabilityDistribution(atoms, weights)
    return probadist
end

end
