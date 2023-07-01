"""
    AbstractLayer

Supertype for all the layers defined in InferOpt.
    
All of these layers are callable, and differentiable with any ChainRules-compatible autodiff backend.

# Interface
- `(layer::AbstractLayer)(args...; kwargs...)`
"""
abstract type AbstractLayer end

## Optimization

"""
    AbstractOptimizationLayer <: AbstractLayer

Supertype for all the optimization layers defined in InferOpt.

# Interface
- `(layer::AbstractOptimizationLayer)(θ; kwargs...)`
- `compute_probability_distribution(layer, θ; kwargs...)` (only if the layer is probabilistic)
"""
abstract type AbstractOptimizationLayer <: AbstractLayer end

## Losses

"""
    AbstractLossLayer <: AbstractLayer

Supertype for all the loss layers defined in InferOpt.

Depending on the precise loss, the arguments to the layer might vary

# Interface
- `(layer::AbstractLossLayer)(θ; kwargs...)` or
- `(layer::AbstractLossLayer)(θ, θ_true; kwargs...)` or
- `(layer::AbstractLossLayer)(θ, y_true; kwargs...)` or
- `(layer::AbstractLossLayer)(θ, (; θ_true, y_true); kwargs...)`
"""
abstract type AbstractLossLayer <: AbstractLayer end

## Checking specific properties

"""
    compute_probability_distribution(layer, θ; kwargs...)

Apply a probabilistic optimization layer to an objective direction `θ` in order to generate a [`FixedAtomsProbabilityDistribution`](@ref) on the vertices of a polytope.
"""
function compute_probability_distribution end
