"""
TODO
"""
struct PerturbedOracle{parallel,D,O,G,R<:AbstractRNG,S<:Union{Nothing,Int}} <:
       AbstractPerturbed{parallel}
    perturbation::D
    oracle::O
    grad_logdensity::G
    rng::R
    seed::S
    nb_samples::Int
end

function PerturbedOracle(
    perturbation::D,
    oracle::O;
    grad_logdensity::G=nothing,
    rng::R=MersenneTwister(0),
    seed::S=nothing,
    is_parallel::Bool=false,
    nb_samples::Int=1,
) where {D,O,G,R<:AbstractRNG,S<:Union{Int,Nothing}}
    return PerturbedOracle{is_parallel,D,O,G,R,S}(
        perturbation, oracle, grad_logdensity, rng, seed, nb_samples
    )
end

function Base.show(io::IO, po::PerturbedOracle)
    (; oracle, perturbation, rng, seed, nb_samples) = po
    return print(
        io, "PerturbedOracle($perturbation, $oracle, $nb_samples, $(typeof(rng)), $seed)"
    )
end

function sample_perturbations(po::PerturbedOracle, θ::AbstractArray)
    (; rng, seed, perturbation, nb_samples) = po
    seed!(rng, seed)
    η_samples = [rand(rng, perturbation(θ)) for _ in 1:nb_samples]
    return η_samples
end

function perturbation_logdensity(po::PerturbedOracle, θ::AbstractArray, η::AbstractArray)
    return logdensityof(po.perturbation(θ), η)
end

function perturbation_grad_logdensity(
    rc::RuleConfig,
    po::PerturbedOracle{parallel,D,O,Nothing},
    θ::AbstractArray,
    η::AbstractArray,
) where {parallel,D,O}
    l, logdensity_pullback = rrule_via_ad(rc, perturbation_logdensity, po, θ, η)
    δperturbation_logdensity, δpo, δθ, δη = logdensity_pullback(one(l))
    return δθ
end

function perturbation_grad_logdensity(
    ::RuleConfig, po::PerturbedOracle, θ::AbstractArray, η::AbstractArray
)
    return po.grad_logdensity(θ, η)
end
