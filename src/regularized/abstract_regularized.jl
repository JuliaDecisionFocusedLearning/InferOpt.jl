"""
    AbstractRegularized{parallel} <: AbstractOptimizationLayer

Convex regularization perturbation of a black box optimizer
```
ŷ(θ) = argmax_{y ∈ C} {θᵀy - Ω(y)}
```

# Interface

- `(regularized::AbstractRegularized)(θ; kwargs...)`: return `ŷ(θ)`
- `compute_regularization(regularized, y)`: return `Ω(y)`

# Available implementations

- [`SoftArgmax`](@ref)
- [`SparseArgmax`](@ref)
- [`RegularizedFrankWolfe`](@ref)
"""
abstract type AbstractRegularized <: AbstractOptimizationLayer end

"""
    compute_regularization(regularized, y)

Return the convex penalty `Ω(y)` associated with an `AbstractRegularized` layer.
"""
function compute_regularization end
