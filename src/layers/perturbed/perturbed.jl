struct Perturbed{R<:Reinforce} <: AbstractOptimizationLayer
    reinforce::R
end

function (perturbed::Perturbed)(θ::AbstractArray)
    return perturbed.reinforce(θ)
end

function is_additive(perturbed::Perturbed)
    return isa(perturbed.reinforce.dist_constructor, AdditivePerturbation)
end

function is_multiplicative(perturbed::Perturbed)
    return isa(perturbed.reinforce.dist_constructor, MultiplicativePerturbation)
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
        Reinforce(
            maximizer, dist_constructor; variance_reduction, seed, threaded, rng, nb_samples
        ),
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
        Reinforce(
            maximizer, dist_constructor; variance_reduction, seed, threaded, rng, nb_samples
        ),
    )
end
