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

function (perturbed_cost::PerturbedCost{F,P})(
    θ::AbstractArray, Z::AbstractArray; kwargs...
) where {F,P<:AbstractPerturbed{F}}
    (; perturbed, cost) = perturbed_cost
    return cost(perturbed(θ, Z; kwargs...); kwargs...)
end

function (perturbed_cost::PerturbedCost{F,P})(
    θ::AbstractArray; kwargs...
) where {F,P<:AbstractPerturbed{F}}
    (; perturbed) = perturbed_cost
    (; M) = perturbed
    Z_samples = [randn(size(θ)) for _ in 1:M]
    return mean(perturbed_cost(θ, Z; kwargs...) for Z in Z_samples)
end
