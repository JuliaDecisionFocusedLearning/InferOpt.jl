"""
    PerturbedOracle{D,F} <: AbstractOptimizationLayer

Differentiable perturbation of a black box optimizer of type `F`, with perturbation of type `D`.

This struct is as wrapper around `Reinforce` from DifferentiableExpectations.jl.

There are three different available constructors that behave differently in the package:
- [`LinearPerturbed`](@ref)
- [`PerturbedAdditive`](@ref)
- [`PerturbedMultiplicative`](@ref)
"""
struct PerturbedOracle{D,F,t,variance_reduction,G,R,S} <: AbstractOptimizationLayer
    reinforce::Reinforce{t,variance_reduction,F,D,G,R,S}
end

function (perturbed::PerturbedOracle)(θ::AbstractArray; kwargs...)
    return perturbed.reinforce(θ; kwargs...)
end

function get_maximizer(perturbed::PerturbedOracle)
    return perturbed.reinforce.f
end

function compute_probability_distribution(
    perturbed::PerturbedOracle, θ::AbstractArray; kwargs...
)
    return empirical_distribution(perturbed.reinforce, θ; kwargs...)
end

function Base.show(io::IO, perturbed::PerturbedOracle{<:AbstractPerturbation})
    (; reinforce) = perturbed
    nb_samples = reinforce.nb_samples
    ε = reinforce.dist_constructor.ε
    seed = reinforce.seed
    rng = reinforce.rng
    perturbation = reinforce.dist_constructor.perturbation_dist
    f = reinforce.f
    return print(
        io,
        "PerturbedOracle($f, ε=$ε, nb_samples=$nb_samples, perturbation=$perturbation, rng=$(typeof(rng)), seed=$seed)",
    )
end

"""
doc
"""
function PerturbedOracle(
    maximizer,
    dist_constructor;
    dist_logdensity_grad=nothing,
    nb_samples=1,
    variance_reduction=true,
    threaded=false,
    seed=nothing,
    rng=Random.default_rng(),
    kwargs...,
)
    return PerturbedOracle(
        Reinforce(
            maximizer,
            dist_constructor,
            dist_logdensity_grad;
            nb_samples,
            variance_reduction,
            threaded,
            seed,
            rng,
            kwargs...,
        ),
    )
end

"""
    PerturbedAdditive(maximizer; kwargs...)
"""
function PerturbedAdditive(
    maximizer;
    ε=1.0,
    perturbation_dist=Normal(0, 1),
    nb_samples=1,
    variance_reduction=true,
    seed=nothing,
    threaded=false,
    rng=Random.default_rng(),
    dist_logdensity_grad=if (perturbation_dist == Normal(0, 1))
        (η, θ) -> ((η .- θ) ./ ε^2,)
    else
        nothing
    end,
)
    dist_constructor = AdditivePerturbation(perturbation_dist, float(ε))
    return PerturbedOracle(
        maximizer,
        dist_constructor;
        dist_logdensity_grad,
        nb_samples,
        variance_reduction,
        seed,
        threaded,
        rng,
    )
end

"""
doc
"""
function PerturbedMultiplicative(
    maximizer;
    ε=1.0,
    perturbation_dist=Normal(0, 1),
    nb_samples=1,
    variance_reduction=true,
    seed=nothing,
    threaded=false,
    rng=Random.default_rng(),
    dist_logdensity_grad=if (perturbation_dist == Normal(0, 1))
        (η, θ) -> (inv.(ε^2 .* θ) .* (η .- θ),)
    else
        nothing
    end,
)
    dist_constructor = MultiplicativePerturbation(perturbation_dist, float(ε))
    return PerturbedOracle(
        maximizer,
        dist_constructor;
        dist_logdensity_grad,
        nb_samples,
        variance_reduction,
        seed,
        threaded,
        rng,
    )
end
