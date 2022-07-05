"""
    RegularizedGeneric{M,RF,RG,F,G,S}

Generic and differentiable regularized prediction function `ŷ(θ) = argmax {θᵀy - Ω(y)}`.

Relies on the Frank-Wolfe algorithm to minimize a concave objective on a polytope.

# Fields
- `maximizer::M`
- `Ω::RF`
- `Ω_grad::RG`
- `f::F`
- `f_grad1::G`
- `linear_solver::S`

See also: [`DifferentiableFrankWolfe`](@ref).
"""
struct RegularizedGeneric{M,RF,RG,F,G,S}
    maximizer::M
    Ω::RF
    Ω_grad::RG
    f::F
    f_grad1::G
    linear_solver::S
end

function Base.show(io::IO, regularized::RegularizedGeneric)
    (; maximizer, Ω, Ω_grad, linear_solver) = regularized
    return print(io, "RegularizedGeneric($maximizer, $Ω, $Ω_grad, $linear_solver)")
end

function RegularizedGeneric(maximizer, Ω, Ω_grad; linear_solver=gmres)
    f(y, θ) = Ω(y) - dot(θ, y)
    f_grad1(y, θ) = Ω_grad(y) - θ
    return RegularizedGeneric(maximizer, Ω, Ω_grad, f, f_grad1, linear_solver)
end

@traitimpl IsRegularized{RegularizedGeneric}

function compute_regularization(regularized::RegularizedGeneric, y::AbstractArray{<:Real})
    return regularized.Ω(y)
end

## Forward pass

function compute_probability_distribution(
    regularized::RegularizedGeneric,
    θ::AbstractArray{<:Real};
    maximizer_kwargs=(;),
    fw_kwargs=(;),
    kwargs...,
)
    (; f, f_grad1, maximizer, linear_solver) = regularized
    lmo = LMOWrapper(maximizer, maximizer_kwargs)
    dfw = DifferentiableFrankWolfe(f, f_grad1, lmo, linear_solver)
    x0 = compute_extreme_point(lmo, θ)
    probadist = compute_probability_distribution(dfw, θ, x0; fw_kwargs=fw_kwargs)
    return probadist
end

function (regularized::RegularizedGeneric)(
    θ::AbstractArray{<:Real}; maximizer_kwargs=(;), fw_kwargs=(;)
)
    probadist = compute_probability_distribution(
        regularized, θ; maximizer_kwargs=maximizer_kwargs, fw_kwargs=fw_kwargs
    )
    return compute_expectation(probadist)
end

## Backward pass, only works with vectors

function ChainRulesCore.rrule(
    rc::RuleConfig,
    ::typeof(compute_probability_distribution),
    regularized::RegularizedGeneric,
    θ::AbstractArray{<:Real};
    maximizer_kwargs=(;),
    fw_kwargs=(;),
    kwargs...,
)
    (; f, f_grad1, maximizer, linear_solver) = regularized
    lmo = LMOWrapper(maximizer, maximizer_kwargs)
    dfw = DifferentiableFrankWolfe(f, f_grad1, lmo, linear_solver)
    x0 = compute_extreme_point(lmo, θ)
    probadist, frank_wolfe_probadist_pullback = rrule(
        rc, compute_probability_distribution, dfw, θ, x0; fw_kwargs=fw_kwargs
    )
    function regularized_generic_probadist_pullback(probadist_tangent)
        _, _, dθ, _ = frank_wolfe_probadist_pullback(probadist_tangent)
        return (NoTangent(), NoTangent(), dθ)
    end
    return probadist, regularized_generic_probadist_pullback
end
