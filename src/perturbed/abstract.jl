"""
    AbstractPerturbed{F}

Differentiable perturbation of a black-box optimizer.

Every concrete subtype must have the following fields:

- `maximizer::F`: underlying argmax function
- `M::Int`: number of noise samples for Monte-Carlo computations

And it must implement the following method:

- [`sample_perturbation(perturbed, θ)`](@ref)
"""
abstract type AbstractPerturbed{F} end

"""
    PerturbedCost{F,P<:AbstractPerturbed{F},C}

Composition of a differentiable perturbed black-box optimizer with an arbitrary cost function. Designed for direct regret minimization (learning by experience).

# Fields
- `perturbed::P`: underlying [`AbstractPerturbed{F}`](@ref) wrapper
- `cost::C`: a real-valued function taking a vector `y` and some `kwargs` as inputs
"""
struct PerturbedCost{F,P<:AbstractPerturbed{F},C}
    perturbed::P
    cost::C
end

"""
    sample_perturbation(perturbed, θ)

Draw a perturbed parameter vector `θs ∼ p_θ(⋅)`
"""
function sample_perturbation(::AbstractPerturbed, ::AbstractArray) end

function (perturbed::AbstractPerturbed)(θ::AbstractArray; kwargs...)
    (; maximizer, M) = perturbed
    θ_samples = [sample_perturbation(perturbed, θ) for _ in 1:M]
    return mean(maximizer(θs; kwargs...) for θs in θ_samples)
end

function (perturbed_cost::PerturbedCost)(θ::AbstractArray; kwargs...)
    (; perturbed, cost) = perturbed_cost
    (; maximizer, M) = perturbed
    θ_samples = [sample_perturbation(perturbed, θ) for _ in 1:M]
    return mean(cost(maximizer(θs; kwargs...); kwargs...) for θs in θ_samples)
end

function compute_y_and_Fθ(perturbed::AbstractPerturbed, θ::AbstractArray; kwargs...)
    (; maximizer, M) = perturbed
    θ_samples = [sample_perturbation(perturbed, θ) for _ in 1:M]
    y_samples = [maximizer(θs; kwargs...) for θs in θ_samples]
    Fθ_samples = [dot(θs, ys) for (θs, ys) in zip(θ_samples, y_samples)]
    return mean(y_samples), mean(Fθ_samples)
end
