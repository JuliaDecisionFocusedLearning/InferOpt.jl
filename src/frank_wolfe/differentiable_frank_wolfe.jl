# The following code is partially adapted from ImplicitDifferentiation.jl

"""
    DifferentiableFrankWolfe{F,G,M,S}

Parameterized version of the Frank-Wolfe algorithm `θ -> argmin_{x ∈ C} f(x, θ)`, which can be differentiated implicitly wrt `θ`.

# Fields
- `f::F`: function `f(x, θ)` to minimize wrt `x`
- `f_grad1::G`: gradient `∇ₓf(x, θ)` of `f` wrt `x`
- `lmo::M`: linear minimization oracle `θ -> argmin_{x ∈ C} θᵀx`, implicitly defines the polytope `C`
- `linear_solver::S`: solver for linear systems of equations, used during implicit differentiation

# Applicable methods

- [`compute_probability_distribution(dfw::DifferentiableFrankWolfe, θ, x0)`](@ref)
- `(dfw::DifferentiableFrankWolfe)(θ, x0)`

"""
struct DifferentiableFrankWolfe{F,G,M<:LinearMinimizationOracle,S}
    f::F
    f_grad1::G
    lmo::M
    linear_solver::S
end

function DifferentiableFrankWolfe(f, f_grad1, lmo, linear_solver=gmres)
    return DifferentiableFrankWolfe(f, f_grad1, lmo, linear_solver)
end

struct SolverFailureException{S} <: Exception
    msg::String
    stats::S
end

function Base.show(io::IO, sfe::SolverFailureException)
    return println(io, "SolverFailureException: $(sfe.msg)\nSolver stats: $(sfe.stats)")
end

## Forward pass

"""
    compute_probability_distribution(dfw::DifferentiableFrankWolfe, θ, x0[; fw_kwargs=(;)])

Compute the optimal active set by applying the away-step Frank-Wolfe algorithm with initial point `x0`, then turn it into a probability distribution.

The named tuple `fw_kwargs` is passed as keyword arguments to `FrankWolfe.away_frank_wolfe`.
"""
function compute_probability_distribution(
    dfw::DifferentiableFrankWolfe,
    θ::AbstractArray{<:Real},
    x0::AbstractArray{<:Real};
    fw_kwargs=(;),
    kwargs...,
)
    (; f, f_grad1, lmo) = dfw
    obj(x) = f(x, θ)
    grad!(g, x) = g .= f_grad1(x, θ)
    full_fw_kwargs = merge(DEFAULT_FRANK_WOLFE_KWARGS, fw_kwargs)
    x, v, primal, dual_gap, traj_data, active_set = away_frank_wolfe(
        obj, grad!, lmo, x0; full_fw_kwargs...
    )
    probadist = FixedAtomsProbabilityDistribution(active_set)
    @assert length(probadist) > 0
    return probadist
end

"""
    (dfw::DifferentiableFrankWolfe)(θ, x0[; fw_kwargs=(;)])

Apply `compute_probability_distribution(dfw, θ, x0)` and return the expectation.
"""
function (dfw::DifferentiableFrankWolfe)(
    θ::AbstractArray{<:Real}, x0::AbstractArray{<:Real}; fw_kwargs=(;), kwargs...
)
    probadist = compute_probability_distribution(dfw, θ, x0; fw_kwargs=fw_kwargs)
    return compute_expectation(probadist)
end

## Backward pass

function frank_wolfe_optimality_conditions(
    dfw::DifferentiableFrankWolfe,
    θ::AbstractArray{<:Real},
    p::AbstractVector{<:Real},
    V::AbstractVector{<:AbstractArray{<:Real}},
)
    (; f_grad1) = dfw
    x = sum(pᵢ * Vᵢ for (pᵢ, Vᵢ) in zip(p, V))
    ∇ₓf = f_grad1(x, θ)
    ∇ₚg = [dot(Vᵢ, ∇ₓf) for Vᵢ in V]
    T = sparse_argmax(p - ∇ₚg)
    return T - p
end

function ChainRulesCore.rrule(
    rc::RuleConfig,
    ::typeof(compute_probability_distribution),
    dfw::DifferentiableFrankWolfe,
    θ::AbstractArray{R1},
    x0::AbstractArray{R2};
    fw_kwargs=(;),
    kwargs...,
) where {R1<:Real,R2<:Real}
    R = promote_type(float(R1), float(R2))
    (; linear_solver) = dfw

    probadist = compute_probability_distribution(dfw, θ, x0; fw_kwargs=fw_kwargs)
    V = probadist.atoms
    p = probadist.weights

    conditions_θ(θ_bis) = frank_wolfe_optimality_conditions(dfw, θ_bis, p, V)
    conditions_p(p_bis) = -frank_wolfe_optimality_conditions(dfw, θ, p_bis, V)

    pullback_Aᵀ = last ∘ rrule_via_ad(rc, conditions_p, p)[2]
    pullback_Bᵀ = last ∘ rrule_via_ad(rc, conditions_θ, θ)[2]

    mul_Aᵀ!(res::Vector, u::Vector) = res .= vec(pullback_Aᵀ(u))
    mul_Bᵀ!(res::Vector, v::Vector) = res .= vec(pullback_Bᵀ(v))

    n, m = length(θ), length(p)
    Aᵀ = LinearOperator(R, m, m, false, false, mul_Aᵀ!)
    Bᵀ = LinearOperator(R, n, m, false, false, mul_Bᵀ!)

    function frank_wolfe_probadist_pullback(probadist_tangent)
        weights_tangent = probadist_tangent.weights
        dp = convert(Vector{R}, unthunk(weights_tangent))
        u, stats = linear_solver(Aᵀ, dp)
        if !stats.solved
            throw(SolverFailureException("Linear solver failed to converge", stats))
        end
        dθ_vec = Bᵀ * u
        dθ = reshape(dθ_vec, size(θ))
        return (NoTangent(), NoTangent(), dθ, NoTangent())
    end

    return probadist, frank_wolfe_probadist_pullback
end
