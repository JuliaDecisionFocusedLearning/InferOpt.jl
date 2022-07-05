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

function FixedAtomsProbabilityDistribution(s::ActiveSet)
    return FixedAtomsProbabilityDistribution(s.atoms, s.weights)
end

Base.length(probadist::FixedAtomsProbabilityDistribution) = length(probadist.atoms)

function Base.rand(rng::AbstractRNG, probadist::FixedAtomsProbabilityDistribution)
    (; atoms, weights) = probadist
    return sample(rng, atoms, StatsBase.Weights(weights))
end

Base.rand(probadist::FixedAtomsProbabilityDistribution) = rand(GLOBAL_RNG, probadist)

function compress!(probadist::FixedAtomsProbabilityDistribution{A,W}; atol=0) where {A,W}
    (; atoms, weights) = probadist
    to_delete = Int[]
    for i in length(probadist):-1:1
        ai = atoms[i]
        for j in 1:(i - 1)
            aj = atoms[j]
            if isapprox(ai, aj; atol=atol)
                weights[j] += weights[i]
                push!(to_delete, i)
                break
            end
        end
    end
    sort!(to_delete)
    deleteat!(atoms, to_delete)
    deleteat!(weights, to_delete)
    return probadist
end

function compute_expectation(
    probadist::FixedAtomsProbabilityDistribution, post_processing=identity; kwargs...
)
    (; atoms, weights) = probadist
    return sum(w * post_processing(a; kwargs...) for (w, a) in zip(weights, atoms))
end

function apply_on_atoms(
    post_processing, probadist::FixedAtomsProbabilityDistribution; kwargs...
)
    (; atoms, weights) = probadist
    post_processed_atoms = [post_processing(a; kwargs...) for a in atoms]
    return FixedAtomsProbabilityDistribution(post_processed_atoms, weights)
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
