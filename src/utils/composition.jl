"""
    ProbabilisticComposition{L,G}

Composition of a probabilistic layer with an arbitrary function (e.g. a cost).

Can be used for direct regret minimization (learning by experience).

# Fields
- `layer::L`: regularized predictor compatible with [`compute_probability_distribution`](@ref)
- `post_processing::P`: function taking an array and some `kwargs` as inputs
"""
struct ProbabilisticComposition{L,P}
    layer::L
    post_processing::P
end

function Base.show(io::IO, composition::ProbabilisticComposition)
    (; layer, post_processing) = composition
    return print(io, "ProbabilisticComposition($layer, $post_processing)")
end

function compute_probability_distribution(composition::ProbabilisticComposition; kwargs...)
    (; layer, post_processing) = composition
    probadist = compute_probability_distribution(layer, θ)
    post_processed_probadist = apply_on_atoms(post_processing, probadist; kwargs...)
    return post_processed_probadist
end

function (composition::ProbabilisticComposition)(θ::AbstractArray{<:Real}; kwargs...)
    (; layer, post_processing) = composition
    probadist = compute_probability_distribution(layer, θ)
    return compute_expectation(probadist, post_processing; kwargs...)
end
