"""
    FixedAtomsProbabilityDistribution{A,W}

Encodes a probability distribution with finite support and fixed atoms.

See [`compute_expectation`](@ref) to understand the name of this struct.

# Fields
- `atoms::Vector{A}`: elements of the support
- `weights::Vector{W}`: probability values for each atom (must sum to 1)
"""
struct FixedAtomsProbabilityDistribution{A,W}
    atoms::Vector{A}
    weights::Vector{W}

    function FixedAtomsProbabilityDistribution(
        atoms::Vector{A}, weights::Vector{W}
    ) where {A,W}
        @assert length(atoms) == length(weights) > 0
        @assert isapprox(sum(weights), one(W); atol=1e-4)
        return new{A,W}(atoms, weights)
    end
end

Base.length(probadist::FixedAtomsProbabilityDistribution) = length(probadist.atoms)

"""
    rand([rng,] probadist)

Sample from the atoms of `probadist` according to their weights.
"""
function Base.rand(rng::AbstractRNG, probadist::FixedAtomsProbabilityDistribution)
    (; atoms, weights) = probadist
    return sample(rng, atoms, StatsBase.Weights(weights))
end

Base.rand(probadist::FixedAtomsProbabilityDistribution) = rand(GLOBAL_RNG, probadist)

"""
    apply_on_atoms(post_processing, probadist)

Create a new distribution by applying the function `post_processing` to each atom of `probadist` (the weights remain the same).
"""
function apply_on_atoms(
    post_processing, probadist::FixedAtomsProbabilityDistribution; kwargs...
)
    (; atoms, weights) = probadist
    post_processed_atoms = [post_processing(a; kwargs...) for a in atoms]
    return FixedAtomsProbabilityDistribution(post_processed_atoms, weights)
end

"""
    compute_expectation(probadist[, post_processing=identity])

Compute the expectation of `post_processing(X)` where `X` is a random variable distributed according to `probadist`.

This operation is made differentiable thanks to a custom reverse rule, even when `post_processing` itself is not a differentiable function.

!!! warning "Warning"
    Derivatives are computed with respect to `probadist.weights` only, assuming that `probadist.atoms` doesn't change (hence the name [`FixedAtomsProbabilityDistribution`](@ref)).
"""
function compute_expectation(
    probadist::FixedAtomsProbabilityDistribution, post_processing=identity; kwargs...
)
    (; atoms, weights) = probadist
    return sum(w * post_processing(a; kwargs...) for (w, a) in zip(weights, atoms))
end

function ChainRulesCore.rrule(
    ::typeof(compute_expectation),
    probadist::FixedAtomsProbabilityDistribution,
    post_processing=identity;
    kwargs...,
)
    e = compute_expectation(probadist, post_processing; kwargs...)
    function expectation_pullback(de)
        d_atoms = NoTangent()
        d_weights = [dot(de, post_processing(a; kwargs...)) for a in probadist.atoms]
        d_probadist = Tangent{FixedAtomsProbabilityDistribution}(;
            atoms=d_atoms, weights=d_weights
        )
        return NoTangent(), d_probadist, NoTangent()
    end
    return e, expectation_pullback
end
