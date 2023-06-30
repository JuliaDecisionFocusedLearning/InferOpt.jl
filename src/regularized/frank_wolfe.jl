"""
    RegularizedFrankWolfe <: AbstractRegularized

Differentiable regularized layer `ŷ(θ) = argmax_{y ∈ C} {θᵀy - Ω(y)}` which relies on the Frank-Wolfe algorithm to define a probability distribution.

!!! warning "Warning"
    Since this is a conditional dependency, you need to have loaded the package DifferentiableFrankWolfe.jl before using `RegularizedFrankWolfe`.

# Fields

- `linear_maximizer`: linear maximization oracle `θ -> argmax_{x ∈ C} θᵀx`, implicitly defines the polytope `C`
- `Ω`: regularization function `Ω(y)`
- `Ω_grad`: gradient function of the regularization function `∇Ω(y)`
- `frank_wolfe_kwargs`: named tuple of keyword arguments passed to the Frank-Wolfe algorithm

# Frank-Wolfe parameters

Some values you can tune:
- `epsilon::Float64`: precision target
- `max_iteration::Integer`: max number of iterations
- `timeout::Float64`: max runtime in seconds
- `lazy::Bool`: caching strategy
- `away_steps::Bool`: avoid zig-zagging
- `line_search::FrankWolfe.LineSearchMethod`: step size selection
- `verbose::Bool`: console output

See the documentation of FrankWolfe.jl for details.
"""
struct RegularizedFrankWolfe{M,RF,RG,FWK} <: AbstractRegularized
    linear_maximizer::M
    Ω::RF
    Ω_grad::RG
    frank_wolfe_kwargs::FWK
end

function RegularizedFrankWolfe(linear_maximizer; Ω, Ω_grad, frank_wolfe_kwargs=NamedTuple())
    return RegularizedFrankWolfe(linear_maximizer, Ω, Ω_grad, frank_wolfe_kwargs)
end

function Base.show(io::IO, regularized::RegularizedFrankWolfe)
    (; linear_maximizer, Ω, Ω_grad, frank_wolfe_kwargs) = regularized
    return print(
        io, "RegularizedFrankWolfe($linear_maximizer, $Ω, $Ω_grad, $frank_wolfe_kwargs)"
    )
end

function compute_regularization(regularized::RegularizedFrankWolfe, y::AbstractArray)
    return regularized.Ω(y)
end

"""
    (regularized::RegularizedFrankWolfe)(θ; kwargs...)

Apply `compute_probability_distribution(regularized, θ)` and return the expectation.

Keyword arguments are passed to the underlying linear maximizer.
"""
function (regularized::RegularizedFrankWolfe)(θ::AbstractArray; kwargs...)
    probadist = compute_probability_distribution(regularized, θ; kwargs...)
    return compute_expectation(probadist)
end
