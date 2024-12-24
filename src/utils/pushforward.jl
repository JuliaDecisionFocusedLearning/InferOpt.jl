"""
$TYPEDEF

Differentiable pushforward of a probabilistic optimization layer with an arbitrary function post-processing function.

`Pushforward` can be used for direct regret minimization (aka learning by experience) when the post-processing returns a cost.

# Fields
$TYPEDFIELDS
"""
struct Pushforward{O<:AbstractOptimizationLayer,P} <: AbstractLayer
    "probabilistic optimization layer"
    optimization_layer::O
    "callable"
    post_processing::P
end

function Base.show(io::IO, pushforward::Pushforward)
    (; optimization_layer, post_processing) = pushforward
    return print(io, "Pushforward($optimization_layer, $post_processing)")
end

"""
$TYPEDSIGNATURES

Output the expectation of `pushforward.post_processing(X)`, where `X` follows the distribution defined by `pushforward.optimization_layer` applied to `θ`.

This function is differentiable, even if `pushforward.post_processing` isn't.
"""
function (pushforward::Pushforward)(θ::AbstractArray; kwargs...)
    (; optimization_layer, post_processing) = pushforward
    probadist = compute_probability_distribution(optimization_layer, θ; kwargs...)
    post_processing_kw = FixKwargs(post_processing, kwargs)
    return mean(post_processing_kw, probadist)
end
