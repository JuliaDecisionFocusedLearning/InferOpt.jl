"""
    RegularizedGeneric{M,RF,RG,F,G,S}

Generic and differentiable regularized prediction function `ŷ(θ) = argmax {θᵀy - Ω(y)}`.

Relies on the Frank-Wolfe algorithm to minimize a concave objective on a polytope.

# Fields
- `maximizer::M`
- `Ω::RF`
- `∇Ω::RG`
- `f::F`
- `∇ₓf::G`
- `linear_solver::S`

See also: [`DifferentiableFrankWolfe`](@ref).
"""
struct RegularizedGeneric{M,RF,RG,F,G,S}
    maximizer::M
    Ω::RF
    ∇Ω::RG
    f::F
    ∇ₓf::G
    linear_solver::S
end

function Base.show(io::IO, regularized::RegularizedGeneric)
    (; maximizer, Ω, ∇Ω, linear_solver) = regularized
    return print(io, "RegularizedGeneric($maximizer, $Ω, $∇Ω, $linear_solver)")
end

function RegularizedGeneric(maximizer, Ω, ∇Ω; linear_solver=gmres)
    f(y, θ) = Ω(y) - dot(θ, y)
    ∇ₓf(y, θ) = ∇Ω(y) - θ
    return RegularizedGeneric(maximizer, Ω, ∇Ω, f, ∇ₓf, linear_solver)
end

@traitimpl IsRegularized{RegularizedGeneric}

function compute_regularization(regularized::RegularizedGeneric, y::AbstractArray{<:Real})
    return regularized.Ω(y)
end

## Forward pass

function optimal_active_set(
    regularized::RegularizedGeneric,
    θ::AbstractArray{<:Real};
    maximizer_kwargs=(;),
    fw_kwargs=(;),
)
    (; f, ∇ₓf, maximizer, linear_solver) = regularized
    lmo = LMOWrapper(maximizer, maximizer_kwargs)
    dfw = DifferentiableFrankWolfe(f, ∇ₓf, lmo, linear_solver)
    x0 = compute_extreme_point(lmo, θ)
    return optimal_active_set(dfw, θ, x0; fw_kwargs=fw_kwargs)
end

function (regularized::RegularizedGeneric)(
    θ::AbstractArray{<:Real}; maximizer_kwargs=(;), fw_kwargs=(;)
)
    active_set = optimal_active_set(
        regularized, θ; maximizer_kwargs=maximizer_kwargs, fw_kwargs=fw_kwargs
    )
    return active_set.x
end

## Backward pass, only works with vectors

function ChainRulesCore.rrule(
    rc::RuleConfig,
    regularized::RegularizedGeneric,
    θ::AbstractArray{<:Real};
    maximizer_kwargs=(;),
    fw_kwargs=(;),
)
    (; f, ∇ₓf, maximizer, linear_solver) = regularized
    lmo = LMOWrapper(maximizer, maximizer_kwargs)
    dfw = DifferentiableFrankWolfe(f, ∇ₓf, lmo, linear_solver)
    x0 = compute_extreme_point(lmo, θ)
    x, frank_wolfe_pullback = rrule(rc, dfw, θ, x0; fw_kwargs=fw_kwargs)
    regularized_generic_pullback(dx) = frank_wolfe_pullback(dx)[1:2]
    return x, regularized_generic_pullback
end
