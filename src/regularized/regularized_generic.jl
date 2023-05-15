"""
    RegularizedGeneric{M,RF,RG}

Differentiable regularized prediction function `ŷ(θ) = argmax_{y ∈ C} {θᵀy - Ω(y)}`.

Relies on the Frank-Wolfe algorithm to minimize a concave objective on a polytope.

# Fields
- `maximizer::M`: linear maximization oracle `θ -> argmax_{x ∈ C} θᵀx`, implicitly defines the polytope `C`
- `Ω::RF`: regularization function `Ω(y)`
- `Ω_grad::RG`: gradient of the regularization function `∇Ω(y)`

# Applicable methods

- [`compute_probability_distribution(regularized::RegularizedGeneric, θ)`](@ref)
- `(regularized::RegularizedGeneric)(θ)`
"""
struct RegularizedGeneric{M,RF,RG}
    maximizer::M
    Ω::RF
    Ω_grad::RG
end

function Base.show(io::IO, regularized::RegularizedGeneric)
    (; maximizer, Ω, Ω_grad) = regularized
    return print(io, "RegularizedGeneric($maximizer, $Ω, $Ω_grad)")
end

@traitimpl IsRegularized{RegularizedGeneric}

function compute_regularization(regularized::RegularizedGeneric, y::AbstractArray{<:Real})
    return regularized.Ω(y)
end

## Forward pass

function compute_probability_distribution(
    dfw::DiffFW, θ::AbstractArray{<:Real}; frank_wolfe_kwargs=NamedTuple()
)
    weights, atoms = dfw.implicit(θ; frank_wolfe_kwargs=frank_wolfe_kwargs)
    probadist = FixedAtomsProbabilityDistribution(atoms, weights)
    return probadist
end

"""
    compute_probability_distribution(
        regularized::RegularizedGeneric, θ;
        maximizer_kwargs=(;), frank_wolfe_kwargs=(;)
    )

Construct a `DifferentiableFrankWolfe.DiffFW` struct and call `compute_probability_distribution` on it.

- The named tuple `maximizer_kwargs` is passed as keyword arguments to the underlying maximizer.
- The named tuple `frank_wolfe_kwargs` is passed as keyword arguments to the underlying Frank-Wolfe algorithm.
"""
function compute_probability_distribution(
    regularized::RegularizedGeneric,
    θ::AbstractArray{<:Real};
    maximizer_kwargs=NamedTuple(),
    frank_wolfe_kwargs=NamedTuple(),
    kwargs...,
)
    (; maximizer, Ω, Ω_grad) = regularized
    f(y, θ) = Ω(y) - dot(θ, y)
    f_grad1(y, θ) = Ω_grad(y) - θ
    lmo = LinearMaximizationOracle(maximizer, maximizer_kwargs)
    dfw = DiffFW(f, f_grad1, lmo)
    probadist = compute_probability_distribution(
        dfw, θ; frank_wolfe_kwargs=frank_wolfe_kwargs
    )
    return probadist
end

"""
    (regularized::RegularizedGeneric)(θ; maximizer_kwargs=(;), frank_wolfe_kwargs=(;))

Apply `compute_probability_distribution(regularized, θ)` and return the expectation.
"""
function (regularized::RegularizedGeneric)(
    θ::AbstractArray{<:Real};
    maximizer_kwargs=NamedTuple(),
    frank_wolfe_kwargs=NamedTuple(),
    kwargs...,
)
    probadist = compute_probability_distribution(
        regularized,
        θ;
        maximizer_kwargs=maximizer_kwargs,
        frank_wolfe_kwargs=frank_wolfe_kwargs,
    )
    return compute_expectation(probadist)
end
