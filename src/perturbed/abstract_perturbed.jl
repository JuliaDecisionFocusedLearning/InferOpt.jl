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

function get_probability_distribution(
    perturbed::AbstractPerturbed, θ::AbstractArray{<:Real}; atol=0, kwargs...
)
    (; nb_samples) = perturbed
    Z_samples = sample_perturbations(perturbed, θ)
    y_samples = [perturbed(θ, Z; kwargs...) for Z in Z_samples]
    multiplicity = ones(Int, nb_samples)
    to_delete = Int[]
    for i in nb_samples:-1:1
        yi = y_samples[i]
        for j in 1:(i - 1)
            yj = y_samples[j]
            if isapprox(yi, yj; atol=atol)
                multiplicity[j] += 1
                push!(to_delete, i)
                break
            end
        end
    end
    sort!(to_delete)
    deleteat!(y_samples, to_delete)
    deleteat!(multiplicity, to_delete)
    weights = multiplicity ./ sum(multiplicity)
    y_mean = sum(w * a for (w, a) in zip(weights, y_samples))
    return ActiveSet(weights, y_samples, y_mean)
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
