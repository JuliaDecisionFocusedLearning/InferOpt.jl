"""
    RegularizedGeneric{M,RF,RG,F,G,S}

Differentiable regularized prediction function `ŷ(θ) = argmax_{y ∈ C} {θᵀy - Ω(y)}`.

Relies on the Frank-Wolfe algorithm to minimize a concave objective on a polytope.

# Fields
- `maximizer::M`: linear maximization oracle `θ -> argmax_{x ∈ C} θᵀx`, implicitly defines the polytope `C`
- `Ω::RF`: regularization function `Ω(y)`
- `Ω_grad::RG`: gradient of the regularization function `∇Ω(y)`
- `f::F`: objective function `f(x, θ) = Ω(y) - θᵀy` minimized by Frank-Wolfe (computed automatically)
- `f_grad1::G`: gradient of the objective function `∇ₓf(x, θ) = ∇Ω(y) - θ` with respect to `x` (computed automatically)
- `linear_solver::S`: solver for linear systems of equations, used during implicit differentiation

# Applicable methods

- [`compute_probability_distribution(regularized::RegularizedGeneric, θ)`](@ref)
- `(regularized::RegularizedGeneric)(θ)`

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

function RegularizedGeneric(maximizer, Ω, Ω_grad, linear_solver=gmres)
    f(y, θ) = Ω(y) - dot(θ, y)
    f_grad1(y, θ) = Ω_grad(y) - θ
    return RegularizedGeneric(maximizer, Ω, Ω_grad, f, f_grad1, linear_solver)
end

function RegularizedGeneric(maximizer::GeneralizedMaximizer, Ω, Ω_grad, linear_solver=gmres)
    f(y, θ) = Ω(y) - objective_value(maximizer, θ, y)
    f_grad1(y, θ) = Ω_grad(y) - θ
    return RegularizedGeneric(maximizer, Ω, Ω_grad, f, f_grad1, linear_solver)
end

"""
    RegularizedGeneric(maximizer[; Ω, Ω_grad, linear_solver=gmres])

Shorter constructor with defaults.
"""
function RegularizedGeneric(
    maximizer;
    Ω=zero_regularization,
    Ω_grad=zero_gradient,
    omega=nothing,
    omega_grad=nothing,
    linear_solver=gmres,
)
    if isnothing(omega) || isnothing(omega_grad)
        return RegularizedGeneric(maximizer, Ω, Ω_grad, linear_solver)
    else
        return RegularizedGeneric(maximizer, omega, omega_grad, linear_solver)
    end
end

@traitimpl IsRegularized{RegularizedGeneric}

function compute_regularization(regularized::RegularizedGeneric, y::AbstractArray{<:Real})
    return regularized.Ω(y)
end

## Forward pass

"""
    compute_probability_distribution(regularized::RegularizedGeneric, θ[; maximizer_kwargs=(;), fw_kwargs=(;)])

Construct a [`DifferentiableFrankWolfe`](@ref) struct and call `compute_probability_distribution` on it.

The named tuple `maximizer_kwargs` is passed as keyword arguments to the underlying maximizer, which is wrapped inside a [`LMOWrapper`](@ref).
The named tuple `fw_kwargs` is passed as keyword arguments to `FrankWolfe.away_frank_wolfe`.
"""
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

"""
    (regularized::RegularizedGeneric)(θ[; maximizer_kwargs=(;), fw_kwargs=(;)])

Apply `compute_probability_distribution(regularized, θ)` and return the expectation.
"""
function (regularized::RegularizedGeneric)(
    θ::AbstractArray{<:Real}; maximizer_kwargs=(;), fw_kwargs=(;), kwargs...
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
