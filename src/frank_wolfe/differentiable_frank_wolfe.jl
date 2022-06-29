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
    dfw::DifferentiableFrankWolfe,
    θ::AbstractArray{<:Real},
    x0::AbstractArray{<:Real};
    fw_kwargs=(;),
)
    (; f, ∇ₓf, lmo) = dfw
    obj(x) = f(x, θ)
    grad!(g, x) = g .= ∇ₓf(x, θ)
    full_fw_kwargs = merge(DEFAULT_FRANK_WOLFE_KWARGS, fw_kwargs)
    x, v, primal, dual_gap, traj_data, active_set = away_frank_wolfe(
        obj, grad!, lmo, x0; full_fw_kwargs...
    )
    @assert length(active_set.atoms) > 0
    return active_set
end

function (dfw::DifferentiableFrankWolfe)(
    θ::AbstractArray{<:Real}, x0::AbstractArray{<:Real}; fw_kwargs=(;)
)
    active_set::ActiveSet = optimal_active_set(dfw, θ, x0; fw_kwargs=fw_kwargs)
    return active_set.x
end

## Backward pass

function frank_wolfe_optimality_conditions(
    dfw::DifferentiableFrankWolfe,
    θ::AbstractArray{<:Real},
    p::AbstractVector{<:Real},
    A::AbstractVector{<:AbstractArray{<:Real}},
)
    (; ∇ₓf) = dfw
    x = sum(pᵢ * Aᵢ for (pᵢ, Aᵢ) in zip(p, A))
    b = ∇ₓf(x, θ)
    ∇ₚg = [dot(Aᵢ, b) for Aᵢ in A]
    T = sparse_argmax(p - ∇ₚg)
    return T - p
end

function ChainRulesCore.rrule(
    rc::RuleConfig,
    dfw::DifferentiableFrankWolfe,
    θ::AbstractArray{R1},
    x0::AbstractArray{R2};
    fw_kwargs=(;),
) where {R1<:Real,R2<:Real}
    R = promote_type(R1, R2)
    (; linear_solver) = dfw

    active_set::ActiveSet = optimal_active_set(dfw, θ, x0; fw_kwargs=fw_kwargs)
    A = active_set.atoms
    p = active_set.weights
    x = active_set.x

    conditions_θ(θ_bis) = frank_wolfe_optimality_conditions(dfw, θ_bis, p, A)
    conditions_p(p_bis) = -frank_wolfe_optimality_conditions(dfw, θ, p_bis, A)

    pullback_Aᵀ = last ∘ rrule_via_ad(rc, conditions_p, p)[2]
    pullback_Bᵀ = last ∘ rrule_via_ad(rc, conditions_θ, θ)[2]

    mul_Aᵀ!(res, u::AbstractVector) = res .= vec(pullback_Aᵀ(reshape(u, size(p))))
    mul_Bᵀ!(res, v::AbstractVector) = res .= vec(pullback_Bᵀ(reshape(v, size(p))))

    n, m = length(θ), length(p)
    Aᵀ = LinearOperator(R, m, m, false, false, mul_Aᵀ!)
    Bᵀ = LinearOperator(R, n, m, false, false, mul_Bᵀ!)

    function frank_wolfe_pullback(dx)
        dx = unthunk(dx)
        dp = [dot(Aᵢ, dx) for Aᵢ in A]
        u, stats = linear_solver(Aᵀ, dp)
        stats.solved || error("Linear solver failed to converge")
        dθ_vec = Bᵀ * u
        dθ = reshape(dθ_vec, size(θ))
        return (NoTangent(), dθ, NoTangent())
    end

    return x, frank_wolfe_pullback
end
