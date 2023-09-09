module InferOptFrankWolfeExt

using DifferentiableFrankWolfe: DiffFW, LinearMaximizationOracleWithKwargs
using FrankWolfe: FrankWolfe, LinearMinimizationOracle
using ImplicitDifferentiation: IterativeLinearSolver
using InferOpt: InferOpt, RegularizedFrankWolfe, FixedAtomsProbabilityDistribution
using InferOpt: compute_expectation, compute_probability_distribution
using LinearAlgebra: dot

"""
    LinearMaximizationOracleWithKwargs{F,K}

Wraps a linear maximizer as a `FrankWolfe.LinearMinimizationOracle` with a sign switch and predefined keyword arguments.

# Fields
- `maximizer::F`: black box linear maximizer
- `maximizer_kwargs::K`: keyword arguments passed to the maximizer whenever it is called
"""
struct LinearMaximizationOracleWithKwargs{F,K} <: LinearMinimizationOracle
    maximizer::F
    maximizer_kwargs::K
end

function LinearMaximizationOracleWithKwargs(maximizer)
    return LinearMaximizationOracleWithKwargs(maximizer, NamedTuple())
end

"""
    FrankWolfe.compute_extreme_point(lmokw::LinearMaximizationOracleWithKwargs, direction)
"""
function FrankWolfe.compute_extreme_point(
    lmokw::LinearMaximizationOracleWithKwargs, direction; kwargs...
)
    opposite_direction = -direction
    v = lmokw.maximizer(opposite_direction; lmokw.maximizer_kwargs...)
    return v
end

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
