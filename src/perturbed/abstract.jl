"""
    AbstractPerturbed{F}

Differentiable perturbation of a black-box optimizer.

Every concrete subtype must have the following fields:

- `maximizer::F`: underlying argmax function
- `M::Int`: number of noise samples for Monte-Carlo computations

And it must implement the following methods:

- [`sample_perturbation(perturbed, θ)`](@ref)
- [`gradlogpdf_perturbation(perturbed, z)`](@ref)
"""
abstract type AbstractPerturbed{F} end

"""
    sample_perturbation(perturbed, θ)

Draw a noise vector `Z` and return the sample `θs = θ + Z`.
"""
function sample_perturbation(::AbstractPerturbed, ::Real) end

"""
    gradlogpdf_perturbation(perturbed, θs, θ)

Compute the gradient of the negative log-density for the noise vector `Z = θs - θ`.
"""
function gradlogpdf_perturbation(::AbstractPerturbed, ::AbstractArray) end

function (perturbed::AbstractPerturbed)(θ::AbstractArray; kwargs...)
    (; maximizer, M) = perturbed
    θ_samples = [sample_perturbation(perturbed, θ) for _ in 1:M]
    return mean(maximizer(θs; kwargs...) for θs in θ_samples)
end

function compute_y_and_Fθ(perturbed::AbstractPerturbed, θ::AbstractArray; kwargs...)
    (; maximizer, M) = perturbed
    θ_samples = [sample_perturbation(perturbed, θ) for _ in 1:M]
    y_samples = [maximizer(θs; kwargs...) for θs in θ_samples]
    Fθ_samples = [dot(θs, ys) for (θs, ys) in zip(θ_samples, y_samples)]
    return mean(y_samples), mean(Fθ_samples)
end

function ChainRulesCore.rrule(perturbed::AbstractPerturbed, θ::AbstractArray; kwargs...)
    (; maximizer, M) = perturbed
    θ_samples = [sample_perturbation(perturbed, θ) for _ in 1:M]
    y_samples = [maximizer(θs; kwargs...) for θs in θ_samples]
    function perturbed_pullback(dy)
        vjp = mean(
            -dot(dy, ys) .* gradlogpdf_perturbation(perturbed, θs, θ) for
            (θs, ys) in zip(θ_samples, y_samples)
        )
        return NoTangent(), vjp
    end
    return mean(y_samples), perturbed_pullback
end
