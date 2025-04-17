module InferOptFrankWolfeExt

using DifferentiableExpectations:
    DifferentiableExpectations, FixedAtomsProbabilityDistribution
using DifferentiableFrankWolfe: DifferentiableFrankWolfe, DiffFW
using FrankWolfe: LinearMinimizationOracle
using ImplicitDifferentiation: KrylovLinearSolver
using InferOpt: InferOpt, RegularizedFrankWolfe
using LinearAlgebra: dot

"""
    RegularizedFrankWolfe(linear_maximizer; Ω, Ω_grad, frank_wolfe_kwargs=(;), implicit_kwargs=(; linear_solver=KrylovLinearSolver(; verbose=false)))

Construct a `RegularizedFrankWolfe` struct with a linear maximizer and the necessary components for the Frank-Wolfe algorithm.
Set `implicit_kwargs` to `(; linear_solver=KrylovLinearSolver(; verbose=true))` if you want to see the solver potential warnings.
"""
function RegularizedFrankWolfe(
    linear_maximizer;
    Ω,
    Ω_grad,
    frank_wolfe_kwargs=NamedTuple(),
    implicit_kwargs=(; linear_solver=KrylovLinearSolver(; verbose=false)),
)
    return RegularizedFrankWolfe(
        linear_maximizer, Ω, Ω_grad, frank_wolfe_kwargs, implicit_kwargs
    )
end

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
    (; linear_maximizer, Ω, Ω_grad, frank_wolfe_kwargs, implicit_kwargs) = regularized
    f(y, θ) = Ω(y) - dot(θ, y)
    f_grad1(y, θ) = Ω_grad(y) - θ
    maximizer(θ; kwargs...) = linear_maximizer(θ; kwargs...)
    lmo = LinearMaximizationOracleWithKwargs(maximizer, kwargs)
    dfw = DiffFW(f, f_grad1, lmo; implicit_kwargs)
    weights, stats = dfw.implicit(θ, frank_wolfe_kwargs)
    atoms = stats.active_set.atoms  # TODO: make it public in DiffFW
    probadist = FixedAtomsProbabilityDistribution(atoms, weights)
    return probadist
end

end
