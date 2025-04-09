"""
$TYPEDEF

Differentiable perturbation of a black box optimizer of type `F`, with perturbation of type `D`.

This struct is as wrapper around `Reinforce` from DifferentiableExpectations.jl.

There are three different available constructors that behave differently in the package:
- [`PerturbedOracle`](@ref)
- [`PerturbedAdditive`](@ref)
- [`PerturbedMultiplicative`](@ref)
"""
struct PerturbedOracle{D,F,t,variance_reduction,G,R,S} <: AbstractOptimizationLayer
    reinforce::Reinforce{t,variance_reduction,F,D,G,R,S}
end

"""
$TYPEDSIGNATURES

Forward pass of the perturbed optimizer.
"""
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
$TYPEDSIGNATURES

Constructor for [`PerturbedOracle`](@ref).
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
$TYPEDSIGNATURES

Constructor for [`PerturbedOracle`](@ref) with an additive perturbation.
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
        NormalAdditiveGradLogdensity(ε)
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
$TYPEDSIGNATURES

Constructor for [`PerturbedOracle`](@ref) with a multiplicative perturbation.
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
        NormalMultiplicativeGradLogdensity(float(ε))
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
