struct Perturbed{F,D,t,variance_reduction,G,R,S} <: AbstractOptimizationLayer
    reinforce::Reinforce{t,variance_reduction,F,D,G,R,S}
end

function (perturbed::Perturbed)(θ::AbstractArray; kwargs...)
    return perturbed.reinforce(θ; kwargs...)
end

function DifferentiableExpectations.empirical_distribution(
    perturbed::Perturbed, θ::AbstractArray; kwargs...
)
    return empirical_distribution(perturbed.reinforce, θ; kwargs...)
end

function Base.show(io::IO, perturbed::Perturbed)
    (; reinforce) = perturbed
    nb_samples = reinforce.nb_samples
    ε = reinforce.dist_constructor.ε
    seed = reinforce.seed
    rng = reinforce.rng
    perturbation = reinforce.dist_constructor.perturbation_dist
    f = reinforce.f
    return print(
        io,
        "Perturbed($f, ε=$ε, nb_samples=$nb_samples, perturbation=$perturbation, rng=$(typeof(rng)), seed=$seed)",
    )
end

function Perturbed(maximizer, dist_constructor; kwargs...)
    return Perturbed(Reinforce(maximizer, dist_constructor; kwargs...))
end

function PerturbedAdditive(
    maximizer;
    ε=1.0,
    perturbation_dist=Normal(0, 1),
    nb_samples=1,
    variance_reduction=true,
    seed=nothing,
    threaded=false,
    rng=Random.default_rng(),
)
    dist_constructor = AdditivePerturbation(perturbation_dist, float(ε))
    return Perturbed(
        maximizer, dist_constructor; nb_samples, variance_reduction, seed, threaded, rng
    )
end

function PerturbedMultiplicative(
    maximizer;
    ε=1.0,
    perturbation_dist=Normal(0, 1),
    nb_samples=1,
    variance_reduction=true,
    seed=nothing,
    threaded=false,
    rng=Random.default_rng(),
)
    dist_constructor = MultiplicativePerturbation(perturbation_dist, float(ε))
    return Perturbed(
        maximizer, dist_constructor; nb_samples, variance_reduction, seed, threaded, rng
    )
end
