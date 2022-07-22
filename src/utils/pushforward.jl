"""
    Pushforward{L,G}

Differentiable pushforward of a probabilistic `layer` with an arbitrary function `post_processing`.

`Pushforward` can be used for direct regret minimization (aka learning by experience) when the post-processing returns a cost.

# Fields
- `layer::L`: anything that implements `compute_probability_distribution(layer, θ; kwargs...)`
- `post_processing::P`: callable

See also: [`FixedAtomsProbabilityDistribution`](@ref).
"""
struct Pushforward{L,P}
    layer::L
    post_processing::P
end

function Base.show(io::IO, pushforward::Pushforward)
    (; layer, post_processing) = pushforward
    return print(io, "Pushforward($layer, $post_processing)")
end

"""
    compute_probability_distribution(pushforward, θ)

Output the distribution of `pushforward.post_processing(X)`, where `X` follows the distribution defined by `pushforward.layer` applied to `θ`.

This function is not differentiable if `pushforward.post_processing` isn't.

See also: [`apply_on_atoms`](@ref).
"""
function compute_probability_distribution(pushforward::Pushforward, θ; kwargs...)
    (; layer, post_processing) = pushforward
    probadist = compute_probability_distribution(layer, θ; kwargs...)
    post_processed_probadist = apply_on_atoms(post_processing, probadist; kwargs...)
    return post_processed_probadist
end

"""
    (pushforward::Pushforward)(θ)

Output the expectation of `pushforward.post_processing(X)`, where `X` follows the distribution defined by `pushforward.layer` applied to `θ`.

Unlike [`compute_probability_distribution(pushforward, θ)`](@ref), this function is differentiable, even if `pushforward.post_processing` isn't.

See also: [`compute_expectation`](@ref).
"""
function (pushforward::Pushforward)(θ::AbstractArray{<:Real}; kwargs...)
    (; layer, post_processing) = pushforward
    probadist = compute_probability_distribution(layer, θ; kwargs...)
    return compute_expectation(probadist, post_processing; kwargs...)
end
