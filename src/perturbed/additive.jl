"""
    PerturbedAdditive <: AbstractPerturbed

Differentiable normal perturbation of a black-box maximizer: the input undergoes `θ -> θ + εZ` where `Z ∼ N(0, I)`.

Reference: <https://arxiv.org/abs/2002.08676>

See [`AbstractPerturbed`](@ref) for more details.
"""
struct PerturbedAdditive{O,R<:AbstractRNG,S<:Union{Nothing,Int},parallel} <:
       AbstractPerturbed{parallel}
    oracle::O
    ε::Float64
    nb_samples::Int
    rng::R
    seed::S

    function PerturbedAdditive{O,R,S,parallel}(
        oracle::O, ε::Float64, nb_samples::Int, rng::R, seed::S
    ) where {O,R<:AbstractRNG,S<:Union{Nothing,Int},parallel}
        @assert parallel isa Bool
        return new{O,R,S,parallel}(oracle, ε, nb_samples, rng, seed)
    end
end

function Base.show(io::IO, perturbed::PerturbedAdditive)
    (; oracle, ε, rng, seed, nb_samples) = perturbed
    return print(io, "PerturbedAdditive($oracle, $ε, $nb_samples, $(typeof(rng)), $seed)")
end

"""
    PerturbedAdditive(maximizer[; ε=1.0, nb_samples=1])
"""
function PerturbedAdditive(
    oracle::O;
    ε=1.0,
    nb_samples=1,
    rng::R=MersenneTwister(0),
    seed::S=nothing,
    is_parallel=false,
) where {O,R,S}
    return PerturbedAdditive{O,R,S,is_parallel}(oracle, float(ε), nb_samples, rng, seed)
end

function sample_perturbations(perturbed::PerturbedAdditive, θ::AbstractArray)
    (; rng, seed, nb_samples, ε) = perturbed
    seed!(rng, seed)
    return [θ .+ ε .* randn(rng, size(θ)) for _ in 1:nb_samples]
end

function perturbation_grad_logdensity(
    ::RuleConfig, perturbed::AbstractPerturbed, θ::AbstractArray, η::AbstractArray
)
    (; ε) = perturbed
    return (η .- θ) ./ ε
end
