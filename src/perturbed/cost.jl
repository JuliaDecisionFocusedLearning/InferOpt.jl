"""
    PerturbedCost{F,P<:AbstractPerturbed{F},C}

Composition of a differentiable perturbed black-box optimizer with an arbitrary cost function.
Designed for direct regret minimization (learning by experience).

# Fields
- `perturbed::P`: underlying [`AbstractPerturbed{F}`](@ref) wrapper
- `cost::C`: a real-valued function taking a vector `y` and some `kwargs` as inputs
"""
struct PerturbedCost{F,P<:AbstractPerturbed{F},C}
    perturbed::P
    cost::C
end

function (perturbed_cost::PerturbedCost)(θ::AbstractArray; kwargs...)
    (; perturbed, cost) = perturbed_cost
    (; maximizer, M) = perturbed
    θ_samples = [sample_perturbation(perturbed, θ) for _ in 1:M]
    return mean(cost(maximizer(θs; kwargs...); kwargs...) for θs in θ_samples)
end

function ChainRulesCore.rrule(perturbed_cost::PerturbedCost, θ::AbstractArray; kwargs...)
    (; perturbed, cost) = perturbed_cost
    (; maximizer, M) = perturbed
    θ_samples = [sample_perturbation(perturbed, θ) for _ in 1:M]
    y_samples = [maximizer(θs; kwargs...) for θs in θ_samples]
    cost_samples = [cost(ys; kwargs...) for ys in y_samples]
    function perturbed_cost_pullback(dc)
        vjp = mean(
            -(dc * cs) .* gradlogpdf_perturbation(perturbed, θs, θ) for
            (cs, θs) in zip(cost_samples, θ_samples)
        )
        return NoTangent(), vjp, NoTangent()
    end
    return mean(cost_samples), perturbed_cost_pullback
end
