# The following code is partially adapted from ImplicitDifferentiation.jl

"""
    DifferentiableFrankWolfe{F,G,M,S}

Parameterized version of the Frank-Wolfe algorithm `θ -> argmin_{x ∈ C} f(x, θ)`.

Compatible with implicit differentiation.

# Fields
- `f::F`: function `f(x, θ)` to minimize wrt `x`
- `∇ₓf::G`: gradient `∇ₓf(x, θ)` of `f` wrt `x`
- `lmo::M`: linear minimization oracle `θ -> argmin_{x ∈ C} θᵀx`
- `linear_solver::S`: solver for linear systems of equations
"""
struct DifferentiableFrankWolfe{F,G,M<:LinearMinimizationOracle,S}
    f::F
    ∇ₓf::G
    lmo::M
    linear_solver::S
end

function DifferentiableFrankWolfe(f, ∇ₓf, lmo; linear_solver=gmres)
    return DifferentiableFrankWolfe(f, ∇ₓf, lmo, linear_solver)
end

## Forward pass

function optimal_active_set(
    dfw::DifferentiableFrankWolfe, θ::AbstractArray{<:Real}; fw_kwargs=(;)
)
    (; f, ∇ₓf, lmo) = dfw
    obj(x) = f(x, θ)
    grad!(g, x) = g .= ∇ₓf(x, θ)
    x0 = compute_extreme_point(lmo, zero(θ))
    full_fw_kwargs = merge(DEFAULT_FRANK_WOLFE_KWARGS, fw_kwargs)
    x, v, primal, dual_gap, traj_data, active_set = away_frank_wolfe(
        obj, grad!, lmo, x0; full_fw_kwargs...
    )
    @assert length(active_set.atoms) > 0
    return active_set
end

function (dfw::DifferentiableFrankWolfe)(θ::AbstractArray{<:Real}; fw_kwargs=(;))
    active_set::ActiveSet = optimal_active_set(dfw, θ; fw_kwargs=fw_kwargs)
    return active_set.x
end

## Backward pass, only works with vectors

function frank_wolfe_optimality_conditions(
    dfw::DifferentiableFrankWolfe,
    θ::AbstractVector{<:Real},
    p::AbstractVector{<:Real},
    V::AbstractMatrix{<:Real},
)
    (; ∇ₓf) = dfw
    ∇ₚg = V' * ∇ₓf(V * p, θ)
    T = sparse_argmax(p - ∇ₚg)
    return T - p
end

function ChainRulesCore.rrule(
    rc::RuleConfig, dfw::DifferentiableFrankWolfe, θ::AbstractVector; fw_kwargs=(;)
)
    (; linear_solver) = dfw

    active_set::ActiveSet = optimal_active_set(dfw, θ; fw_kwargs=fw_kwargs)
    V = reduce(hcat, active_set.atoms)
    p = active_set.weights
    n, m = length(θ), length(p)

    conditions_θ(θ_bis) = frank_wolfe_optimality_conditions(dfw, θ_bis, p, V)
    conditions_p(p_bis) = -frank_wolfe_optimality_conditions(dfw, θ, p_bis, V)

    pullback_Aᵀ = last ∘ rrule_via_ad(rc, conditions_p, p)[2]
    pullback_Bᵀ = last ∘ rrule_via_ad(rc, conditions_θ, θ)[2]

    mul_Aᵀ!(res, v) = res .= pullback_Aᵀ(v)
    mul_Bᵀ!(res, v) = res .= pullback_Bᵀ(v)

    Aᵀ = LinearOperator(Float64, m, m, false, false, mul_Aᵀ!)
    Bᵀ = LinearOperator(Float64, n, m, false, false, mul_Bᵀ!)

    function frank_wolfe_pullback(dx)
        dp = V' * Vector(unthunk(dx))
        u, stats = linear_solver(Aᵀ, dp)
        if !stats.solved
            error("The linear solver failed to converge")
        end
        dθ = Bᵀ * u
        return (NoTangent(), dθ)
    end

    return active_set.x, frank_wolfe_pullback
end
