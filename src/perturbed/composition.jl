"""
    PerturbedComposition{F,P<:AbstractPerturbed{F},G}

Composition of a differentiable perturbed black-box optimizer with an arbitrary function.

Suitable for direct regret minimization (learning by experience) when said function is a cost.

# Fields
- `perturbed::P`: underlying [`AbstractPerturbed{F}`](@ref) wrapper
- `g::G`: function taking an array `y` and some `kwargs` as inputs

The method [`rrule(perturbed_composition, θ; kwargs...)`](@ref) must be implemented individually for each specific type `P`.
"""
struct PerturbedComposition{F,P<:AbstractPerturbed{F},G}
    perturbed::P
    g::G
end

function Base.show(io::IO, perturbed_composition::PerturbedComposition)
    (; perturbed, g) = perturbed_composition
    return print(io, "PerturbedComposition($perturbed, $g)")
end

"""
    ∘(g, perturbed)

Create a `PerturbedComposition` object from `perturbed` and `g`.
"""
Base.:∘(g, perturbed::AbstractPerturbed) = PerturbedComposition(perturbed, g)

function (perturbed_composition::PerturbedComposition{F,P})(
    θ::AbstractArray{<:Real}; kwargs...
) where {F,P<:AbstractPerturbed{F}}
    (; perturbed, g) = perturbed_composition
    Z_samples = sample_perturbations(perturbed, θ)
    g_samples = [g(perturbed(θ, Z; kwargs...); kwargs...) for Z in Z_samples]
    return mean(g_samples)
end

"""
    rrule(perturbed_composition, θ; kwargs...)
"""
function ChainRulesCore.rrule(
    perturbed_composition::PerturbedComposition, θ::AbstractArray{<:Real}; kwargs...
)
    return error("not implemented")
end
