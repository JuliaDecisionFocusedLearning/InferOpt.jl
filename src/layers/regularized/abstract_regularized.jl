"""
    AbstractRegularized <: AbstractOptimizationLayer

Convex regularization perturbation of a black box linear (in θ) optimizer
```
ŷ(θ) = argmax_{y ∈ C} {θᵀg(y) + h(y) - Ω(y)}
```
with g and h functions of y.

# Interface
- `(regularized::AbstractRegularized)(θ; kwargs...)`: return `ŷ(θ)`
- `compute_regularization(regularized, y)`: return `Ω(y)
- `get_maximizer(regularized)`: return the associated optimizer

# Available implementations
- [`SoftArgmax`](@ref)
- [`SparseArgmax`](@ref)
- [`SoftRank`](@ref)
- [`RegularizedFrankWolfe`](@ref)
"""
abstract type AbstractRegularized <: AbstractOptimizationLayer end

"""
    compute_regularization(regularized::AbstractRegularized, y)

Return the convex penalty `Ω(y)` associated with an `AbstractRegularized` layer.
"""
function compute_regularization end

@required AbstractRegularized begin
    # (regularized::AbstractRegularized)(θ::AbstractArray; kwargs...) # waiting for RequiredInterfaces to support this (see https://github.com/Seelengrab/RequiredInterfaces.jl/issues/11)
    compute_regularization(::AbstractRegularized, y)
end
