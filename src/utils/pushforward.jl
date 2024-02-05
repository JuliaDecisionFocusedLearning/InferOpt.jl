"""
$TYPEDEF

Differentiable pushforward of a probabilistic optimization layer with an arbitrary function post-processing function.

`Pushforward` can be used for direct regret minimization (aka learning by experience) when the post-processing returns a cost.

# Fields
$TYPEDFIELDS

See also: [`FixedAtomsProbabilityDistribution`](@ref).
"""
struct Pushforward{O<:AbstractOptimizationLayer,P} <: AbstractLayer
    "probabilistic optimization layer"
    optimization_layer::O
    "callable post-processing cost"
    post_processing::P
end

function Base.show(io::IO, pushforward::Pushforward)
    (; optimization_layer, post_processing) = pushforward
    return print(io, "Pushforward($optimization_layer, $post_processing)")
end

"""
$TYPEDSIGNATURES

Output the distribution of `pushforward.post_processing(X)`, where `X` follows the distribution defined by `pushforward.optimization_layer` applied to `θ`.

This function is not differentiable if `pushforward.post_processing` isn't.

See also: [`apply_on_atoms`](@ref).
"""
function compute_probability_distribution(pushforward::Pushforward, θ; kwargs...)
    (; optimization_layer, post_processing) = pushforward
    probadist = compute_probability_distribution(optimization_layer, θ; kwargs...)
    post_processed_probadist = apply_on_atoms(post_processing, probadist; kwargs...)
    return post_processed_probadist
end

"""
    (pushforward::Pushforward)(θ; kwargs...)

Output the expectation of `pushforward.post_processing(X)`, where `X` follows the distribution defined by `pushforward.optimization_layer` applied to `θ`.

Unlike [`compute_probability_distribution(pushforward, θ)`](@ref), this function is differentiable, even if `pushforward.post_processing` isn't.

See also: [`compute_expectation`](@ref).
"""
function (pushforward::Pushforward)(θ::AbstractArray; kwargs...)
    (; optimization_layer, post_processing) = pushforward
    probadist = compute_probability_distribution(optimization_layer, θ; kwargs...)
    return compute_expectation(probadist, post_processing; kwargs...)
end
