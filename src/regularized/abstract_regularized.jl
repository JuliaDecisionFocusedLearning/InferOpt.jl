"""
$TYPEDEF

Convex regularization perturbation of a black box linear optimizer
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
$TYPEDEF

Convex regularization perturbation of a black box **generalized** optimizer
```
ŷ(θ) = argmax_{y ∈ C} {θᵀg(y) + h(y) - Ω(y)}
with g and h functions of y.
```

# Interface

- `(regularized::AbstractRegularized)(θ; kwargs...)`: return `ŷ(θ)`
- `compute_regularization(regularized, y)`: return `Ω(y)`
- `get_maximizer(regularized)`: return the associated `GeneralizedMaximizer` optimizer
"""
abstract type AbstractRegularizedGeneralizedMaximizer <: AbstractRegularized end

"""
    $FUNCTIONNAME(regularized, y)

Return the convex penalty `Ω(y)` associated with an `AbstractRegularized` layer.
"""
function compute_regularization end

"""
    $FUNCTIONNAME(regularized)

Return the associated optimizer.
"""
function get_maximizer end

# Required interface

@required AbstractRegularized begin
    # (regularized::AbstractRegularized)(θ::AbstractArray; kwargs...) # waiting for RequiredInterfaces to support this (see https://github.com/Seelengrab/RequiredInterfaces.jl/issues/11)
    compute_regularization(::AbstractRegularized, y)
end

@required AbstractRegularizedGeneralizedMaximizer begin
    get_maximizer(::AbstractRegularizedGeneralizedMaximizer)
end
