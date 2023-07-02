"""
    PerturbedMultiplicative <: AbstractPerturbed

Differentiable log-normal perturbation of a black-box maximizer: the input undergoes `θ -> θ ⊙ exp[εZ - ε²/2]` where `Z ∼ N(0, I)`.

Reference: <https://arxiv.org/abs/2207.13513>

See [`AbstractPerturbed`](@ref) for more details.
"""
struct PerturbedMultiplicative{O,R<:AbstractRNG,S<:Union{Nothing,Int},parallel} <:
       AbstractPerturbed{parallel}
    oracle::O
    ε::Float64
    nb_samples::Int
    rng::R
    seed::S

    function PerturbedMultiplicative{O,R,S,parallel}(
        oracle::O, ε::Float64, nb_samples::Int, rng::R, seed::S
    ) where {O,R<:AbstractRNG,S<:Union{Nothing,Int},parallel}
        @assert parallel isa Bool
        return new{O,R,S,parallel}(oracle, ε, nb_samples, rng, seed)
    end
end

function Base.show(io::IO, perturbed::PerturbedMultiplicative)
    (; oracle, ε, rng, seed, nb_samples) = perturbed
    return print(
        io, "PerturbedMultiplicative($oracle, $ε, $nb_samples, $(typeof(rng)), $seed)"
    )
end

"""
    PerturbedMultiplicative(maximizer[; ε=1.0, nb_samples=1])
"""
function PerturbedMultiplicative(
    oracle::F;
    ε=1.0,
    nb_samples=1,
    rng::R=MersenneTwister(0),
    seed::S=nothing,
    is_parallel=false,
) where {F,R,S}
    return PerturbedMultiplicative{F,R,S,is_parallel}(
        oracle, float(ε), nb_samples, rng, seed
    )
end

function sample_perturbations(perturbed::PerturbedMultiplicative, θ::AbstractArray)
    (; rng, seed, nb_samples, ε) = perturbed
    seed!(rng, seed)
    return [θ .* exp.(ε .* randn(rng, size(θ)) .- ε^2 / 2) for _ in 1:nb_samples]
end

function perturbation_grad_logdensity(
    ::RuleConfig, perturbed::PerturbedMultiplicative, θ::AbstractArray, η::AbstractArray
)
    (; ε) = perturbed
    return inv.(ε .* θ) .* (η .- θ)  # TODO: check formula
end
