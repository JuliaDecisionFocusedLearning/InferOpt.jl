"""
    PerturbedGeneric{F,D}

Differentiable perturbation of a black-box optimizer.

# Fields
- `maximizer::F`: underlying argmax function
- `noise_dist::D`: function taking `θ` and returning a local noise distribution
- `M::Int`: number of noise samples for Monte-Carlo computations
"""
struct PerturbedGeneric{F,D} <: AbstractPerturbed
    maximizer::F
    noise_dist::D
    M::Int
end

function PerturbedGeneric(maximizer::F; noise_dist::D, M) where {F,D}
    return PerturbedGeneric{F,D}(maximizer, noise_dist, M)
end

function (perturbed::PerturbedGeneric)(θ::AbstractArray; kwargs...)
    (; maximizer, noise_dist, M) = perturbed
    local_noise_dist = noise_dist(θ)
    y_samples = Folds.map(m -> maximizer(rand(local_noise_dist); kwargs...), 1:M)
    y_mean = mean(y_samples)
    return y_mean
end

function compute_y_and_Fθ(perturbed::PerturbedGeneric, θ::AbstractArray; kwargs...)
    (; maximizer, noise_dist, M) = perturbed
    local_noise_dist = noise_dist(θ)
    perturbed_θs = [rand(local_noise_dist) for _ in 1:M]
    y_samples = Folds.map(θ_perturbed -> maximizer(θ_perturbed; kwargs...), perturbed_θs)
    F_θ_sample = [dot(θ_perturbed, y) for (θ_perturbed, y) in zip(perturbed_θs, y_samples)]
    y_mean = mean(y_samples)
    Fθ_mean = mean(F_θ_sample)  # useful for computing Fenchel-Young loss
    return y_mean, Fθ_mean
end

function ChainRulesCore.rrule(perturbed::PerturbedGeneric, θ::AbstractArray; kwargs...)
    (; maximizer, noise_dist, M) = perturbed
    local_noise_dist = noise_dist(θ)
    ∇ν(z) = -gradlogpdf(local_noise_dist, z + θ)
    εZ_samples = [rand(local_noise_dist) - θ for m in 1:M]
    y_samples = [maximizer(θ + εZ; kwargs...) for εZ in εZ_samples]
    y_mean = mean(y_samples)

    function perturbed_generic_pullback(dy)
        vjp = mean(dot(dy, y_samples[m]) * ∇ν(εZ_samples[m]) for m in 1:M)
        return NoTangent(), vjp
    end

    return y_mean, perturbed_generic_pullback
end
