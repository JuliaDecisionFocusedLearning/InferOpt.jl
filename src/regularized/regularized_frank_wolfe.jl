"""
$TYPEDEF

Regularized optimization layer which relies on the Frank-Wolfe algorithm to define a probability distribution while solving
```
ŷ(θ) = argmax_{y ∈ C} {θᵀy - Ω(y)}
```

!!! warning "Warning"
    Since this is a conditional dependency, you need to have loaded the package DifferentiableFrankWolfe.jl before using `RegularizedFrankWolfe`.

# Fields
$TYPEDFIELDS

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
    "linear maximization oracle `θ -> argmax_{x ∈ C} θᵀx`, implicitly defines the polytope `C`"
    linear_maximizer::M
    "regularization function `Ω(y)`"
    Ω::RF
    "gradient function of the regularization function `∇Ω(y)`"
    Ω_grad::RG
    "named tuple of keyword arguments passed to the Frank-Wolfe algorithm"
    frank_wolfe_kwargs::FWK
end

"""
    RegularizedFrankWolfe(linear_maximizer; Ω, Ω_grad, frank_wolfe_kwargs=(;))
"""
function RegularizedFrankWolfe(linear_maximizer; Ω, Ω_grad, frank_wolfe_kwargs=NamedTuple())
    return RegularizedFrankWolfe(linear_maximizer, Ω, Ω_grad, frank_wolfe_kwargs)
end

function Base.show(io::IO, regularized::RegularizedFrankWolfe)
    (; linear_maximizer, Ω, Ω_grad, frank_wolfe_kwargs) = regularized
    return print(
        io, "RegularizedFrankWolfe($linear_maximizer, $Ω, $Ω_grad, $frank_wolfe_kwargs)"
    )
end

function compute_regularization(regularized::RegularizedFrankWolfe, y)
    return regularized.Ω(y)
end

"""
    (regularized::RegularizedFrankWolfe)(θ; kwargs...)

Apply `compute_probability_distribution(regularized, θ; kwargs...)` and return the expectation.
"""
function (regularized::RegularizedFrankWolfe)(θ::AbstractArray; kwargs...)
    probadist = compute_probability_distribution(regularized, θ; kwargs...)
    return compute_expectation(probadist)
end
