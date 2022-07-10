"""
    Pushforward{L,G}

Differentiable composition of a probabilistic `layer` with an arbitrary function `post_processing`.

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

function Base.show(io::IO, composition::Pushforward)
    (; layer, post_processing) = composition
    return print(io, "Pushforward($layer, $post_processing)")
end

"""
    compute_probability_distribution(composition, θ)

Output the distribution of `composition.post_processing(X)`, where `X` follows the distribution defined by `composition.layer` applied to `θ`.

This function is not differentiable if `composition.post_processing` isn't.

See also: [`apply_on_atoms`](@ref).
"""
function compute_probability_distribution(composition::Pushforward, θ; kwargs...)
    (; layer, post_processing) = composition
    probadist = compute_probability_distribution(layer, θ; kwargs...)
    post_processed_probadist = apply_on_atoms(post_processing, probadist; kwargs...)
    return post_processed_probadist
end

"""
    (composition::Pushforward)(θ)

Output the expectation of `composition.post_processing(X)`, where `X` follows the distribution defined by `composition.layer` applied to `θ`.

Unlike [`compute_probability_distribution(composition, θ)`](@ref), this function is differentiable, even if `composition.post_processing` isn't.

See also: [`compute_expectation`](@ref).
"""
function (composition::Pushforward)(θ::AbstractArray{<:Real}; kwargs...)
    (; layer, post_processing) = composition
    probadist = compute_probability_distribution(layer, θ; kwargs...)
    return compute_expectation(probadist, post_processing; kwargs...)
end
