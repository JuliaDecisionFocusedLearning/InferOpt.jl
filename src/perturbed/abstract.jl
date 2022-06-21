"""
    AbstractPerturbed{F}

Differentiable perturbation of a black-box optimizer.

# Subtypes
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
- `compute_y_and_F(perturbed, θ, Z; kwargs...)`
"""
abstract type AbstractPerturbed{F} end

function sample_perturbations(perturbed::AbstractPerturbed, θ::AbstractArray{<:Real})
    (; rng, seed, nb_samples) = perturbed
    Random.seed!(rng, seed)
    Z_samples = [randn(rng, size(θ)) for _ in 1:nb_samples]
    return Z_samples
end

function (perturbed::AbstractPerturbed)(θ::AbstractArray{<:Real}; kwargs...)
    Z_samples = sample_perturbations(perturbed, θ)
    y_samples = [perturbed(θ, Z; kwargs...) for Z in Z_samples]
    return mean(y_samples)
end

function compute_y_and_F(perturbed::AbstractPerturbed, θ::AbstractArray{<:Real}; kwargs...)
    Z_samples = sample_perturbations(perturbed, θ)
    y_and_F_samples = [compute_y_and_F(perturbed, θ, Z; kwargs...) for Z in Z_samples]
    return mean(first, y_and_F_samples), mean(last, y_and_F_samples)
end

"""
    PerturbedComposition{F,P<:AbstractPerturbed{F},G}

Composition of a differentiable perturbed black-box optimizer with an arbitrary function.

Suitable for direct regret minimization (learning by experience) when that function is a cost.

# Fields
- `perturbed::P`: underlying [`AbstractPerturbed{F}`](@ref) wrapper
- `g::G`: function taking an array `y` and some `kwargs` as inputs
"""
struct PerturbedComposition{F,P<:AbstractPerturbed{F},G}
    perturbed::P
    g::G
end

function Base.show(io::IO, perturbed_composition::PerturbedComposition)
    (; perturbed, g) = perturbed_composition
    return print(io, "PerturbedComposition($perturbed, $g)")
end

Base.:∘(g, perturbed::AbstractPerturbed) = PerturbedComposition(perturbed, g)

function (perturbed_composition::PerturbedComposition{F,P})(
    θ::AbstractArray{<:Real}; kwargs...
) where {F,P<:AbstractPerturbed{F}}
    (; perturbed, g) = perturbed_composition
    Z_samples = sample_perturbations(perturbed, θ)
    g_samples = [g(perturbed(θ, Z; kwargs...); kwargs...) for Z in Z_samples]
    return mean(g_samples)
end
