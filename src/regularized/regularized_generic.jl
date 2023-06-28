"""
    RegularizedGeneric{M,RF,RG}

Differentiable regularized prediction function `ŷ(θ) = argmax_{y ∈ C} {θᵀy - Ω(y)}`.

Relies on the Frank-Wolfe algorithm to minimize a concave objective on a polytope.

!!! warning "Warning"
    Since this is a conditional dependency, you need to run `import DifferentiableFrankWolfe` before using `RegularizedGeneric`.

# Fields
- `maximizer::M`: linear maximization oracle `θ -> argmax_{x ∈ C} θᵀx`, implicitly defines the polytope `C`
- `Ω::RF`: regularization function `Ω(y)`
- `Ω_grad::RG`: gradient of the regularization function `∇Ω(y)`
- `frank_wolfe_kwargs::FWK`: keyword arguments passed to the Frank-Wolfe algorithm

# Applicable methods

- [`compute_probability_distribution(regularized::RegularizedGeneric, θ; kwargs...)`](@ref)
- `(regularized::RegularizedGeneric)(θ; kwargs...)`

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
struct RegularizedGeneric{M,RF,RG,FWK}
    maximizer::M
    Ω::RF
    Ω_grad::RG
    frank_wolfe_kwargs::FWK
end

function Base.show(io::IO, regularized::RegularizedGeneric)
    (; maximizer, Ω, Ω_grad) = regularized
    return print(io, "RegularizedGeneric($maximizer, $Ω, $Ω_grad)")
end

@traitimpl IsRegularized{RegularizedGeneric}

function compute_regularization(regularized::RegularizedGeneric, y::AbstractArray)
    return regularized.Ω(y)
end

"""
    (regularized::RegularizedGeneric)(θ; kwargs...)

Apply `compute_probability_distribution(regularized, θ, kwargs...)` and return the expectation.
"""
function (regularized::RegularizedGeneric)(θ::AbstractArray; kwargs...)
    probadist = compute_probability_distribution(regularized, θ; kwargs...)
    return compute_expectation(probadist)
end
