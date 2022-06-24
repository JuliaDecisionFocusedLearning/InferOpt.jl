"""
    AbstractPerturbed{F}

Differentiable perturbation of a black-box optimizer.

# Available subtypes
- [`PerturbedAdditive{F}`](@ref)
- [`PerturbedMultiplicative{F}`](@ref)

# Required fields
- `maximizer::F`: underlying argmax function
- `ε::Float64`: noise scaling parameter
- `rng::AbstractRNG`: random number generator
- `seed::Union{Nothing,Int}`: random seed
- `nb_samples::Int`: number of random samples for Monte-Carlo computations

# Required methods
- `(perturbed)(θ, Z; kwargs...)`
- [`compute_y_and_F(perturbed, θ, Z; kwargs...)`](@ref)

# Optional methods
- `rrule(perturbed, θ; kwargs...)`
"""
abstract type AbstractPerturbed{F} end

function sample_perturbations(perturbed::AbstractPerturbed, θ::AbstractArray{<:Real})
    (; rng, seed, nb_samples) = perturbed
    Random.seed!(rng, seed)
    Z_samples = [randn(rng, size(θ)) for _ in 1:nb_samples]
    return Z_samples
end

"""
    (perturbed)(θ, Z; kwargs...)
"""
function (perturbed::AbstractPerturbed)(
    θ::AbstractArray{<:Real}, Z::AbstractArray{<:Real}; kwargs...
)
    return error("not implemented")
end

function (perturbed::AbstractPerturbed)(θ::AbstractArray{<:Real}; kwargs...)
    Z_samples = sample_perturbations(perturbed, θ)
    y_samples = [perturbed(θ, Z; kwargs...) for Z in Z_samples]
    return mean(y_samples)
end

"""
    compute_y_and_F(perturbed, θ, Z; kwargs...)
"""
function compute_y_and_F(
    perturbed::AbstractPerturbed,
    θ::AbstractArray{<:Real},
    Z::AbstractArray{<:Real};
    kwargs...,
)
    return error("not implemented")
end

function compute_y_and_F(perturbed::AbstractPerturbed, θ::AbstractArray{<:Real}; kwargs...)
    Z_samples = sample_perturbations(perturbed, θ)
    y_and_F_samples = [compute_y_and_F(perturbed, θ, Z; kwargs...) for Z in Z_samples]
    return mean(first, y_and_F_samples), mean(last, y_and_F_samples)
end
