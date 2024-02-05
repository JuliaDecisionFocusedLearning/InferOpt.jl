"""
$TYPEDEF

Supertype for all the layers defined in InferOpt.

All of these layers are callable, and differentiable with any ChainRules-compatible autodiff backend.

# Interface
- `(layer::$FUNCTIONNAME)(args...; kwargs...)`
"""
abstract type AbstractLayer end

## Optimization

"""
$TYPEDEF

Supertype for all the optimization layers defined in `InferOpt`.

# Interface
- `(layer::$FUNCTIONNAME)(θ; kwargs...)`
- `compute_probability_distribution(layer::$FUNCTIONNAME, θ; kwargs...)` (only if the layer is probabilistic)
"""
abstract type AbstractOptimizationLayer <: AbstractLayer end

## Losses

"""
$TYPEDEF

Supertype for all the loss layers defined in `InferOpt`.

Depending on the precise loss, the arguments to the layer might vary

# Interface
- `(layer::$FUNCTIONNAME)(θ; kwargs...)` or
- `(layer::$FUNCTIONNAME)(θ, θ_true; kwargs...)` or
- `(layer::$FUNCTIONNAME)(θ, y_true; kwargs...)` or
- `(layer::$FUNCTIONNAME)(θ, (; θ_true, y_true); kwargs...)`
"""
abstract type AbstractLossLayer <: AbstractLayer end

## Checking specific properties

"""
    $FUNCTIONNAME(layer, θ; kwargs...)

Apply a probabilistic optimization layer to an objective direction `θ` in order to generate a [`FixedAtomsProbabilityDistribution`](@ref) on the vertices of a polytope.

# Method list
$METHODLIST
"""
function compute_probability_distribution end
