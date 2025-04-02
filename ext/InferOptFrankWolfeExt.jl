module InferOptFrankWolfeExt

using DifferentiableExpectations:
    DifferentiableExpectations, FixedAtomsProbabilityDistribution
using DifferentiableFrankWolfe: DifferentiableFrankWolfe, DiffFW
using FrankWolfe: LinearMinimizationOracle
using ImplicitDifferentiation: KrylovLinearSolver
using InferOpt: InferOpt, RegularizedFrankWolfe
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

"""
    FrankWolfe.compute_extreme_point(lmokw::LinearMaximizationOracleWithKwargs, direction)
"""
function DifferentiableFrankWolfe.compute_extreme_point(
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
    shape = size(θ)
    (; linear_maximizer, Ω, Ω_grad, frank_wolfe_kwargs) = regularized
    f(y, θ) = Ω(y) - dot(θ, y)
    f_grad1(y, θ) = Ω_grad(y) - θ
    maximizer(θ; shape, kwargs...) = vec(linear_maximizer(reshape(θ, shape); kwargs...))
    lmo = LinearMaximizationOracleWithKwargs(maximizer, (; shape, kwargs...))
    implicit_kwargs = (; linear_solver=KrylovLinearSolver())
    dfw = DiffFW(f, f_grad1, lmo; implicit_kwargs)
    weights, atoms = dfw.implicit(vec(θ); frank_wolfe_kwargs=frank_wolfe_kwargs)
    probadist = FixedAtomsProbabilityDistribution(
        map(atom -> reshape(atom, shape), atoms), weights
    )
    return probadist
end

end
