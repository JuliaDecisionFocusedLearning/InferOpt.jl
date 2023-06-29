"""
    FrankWolfeOptimizer{M,RF,RG,FWK}

Differentiable regularized prediction function `ŷ(θ) = argmax_{y ∈ C} {θᵀy - Ω(y)}`.

Relies on the Frank-Wolfe algorithm to minimize a concave objective on a polytope.

!!! warning "Warning"
    Since this is a conditional dependency, you need to run `import DifferentiableFrankWolfe` before using `RegularizedGeneric`.

# Fields
- `linear_maximizer::M`: linear maximization oracle `θ -> argmax_{x ∈ C} θᵀx`, implicitly defines the polytope `C`
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
struct FrankWolfeOptimizer{M,RF,RG,FWK}
    linear_maximizer::M
    Ω::RF
    Ω_grad::RG
    frank_wolfe_kwargs::FWK
end

function Base.show(io::IO, optimizer::FrankWolfeOptimizer)
    (; linear_maximizer, Ω, Ω_grad) = optimizer
    return print(io, "RegularizedGeneric($linear_maximizer, $Ω, $Ω_grad)")
end
