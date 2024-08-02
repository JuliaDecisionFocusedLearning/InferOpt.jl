"""
    Pushforward <: AbstractLayer

Differentiable pushforward of a probabilistic optimization layer with an arbitrary function post-processing function.

`Pushforward` can be used for direct regret minimization (aka learning by experience) when the post-processing returns a cost.

# Fields
- `optimization_layer::AbstractOptimizationLayer`: probabilistic optimization layer
- `post_processing`: callable

See also: `FixedAtomsProbabilityDistribution`.
"""
struct Pushforward{O<:AbstractOptimizationLayer,P} <: AbstractLayer
    optimization_layer::O
    post_processing::P
end

function Base.show(io::IO, pushforward::Pushforward)
    (; optimization_layer, post_processing) = pushforward
    return print(io, "Pushforward($optimization_layer, $post_processing)")
end

"""
    (pushforward::Pushforward)(θ; kwargs...)

Output the expectation of `pushforward.post_processing(X)`, where `X` follows the distribution defined by `pushforward.optimization_layer` applied to `θ`.

This function is differentiable, even if `pushforward.post_processing` isn't.

See also: `compute_expectation`.
"""
function (pushforward::Pushforward)(θ::AbstractArray; kwargs...)
    (; optimization_layer, post_processing) = pushforward
    probadist = compute_probability_distribution(optimization_layer, θ; kwargs...)
    post_processing_kw = FixKwargs(post_processing, kwargs)
    return mean(post_processing_kw, probadist)
end
