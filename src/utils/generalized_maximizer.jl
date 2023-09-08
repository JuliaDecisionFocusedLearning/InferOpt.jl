"""
    GeneralizedMaximizer{F,G,H}

Wrapper for generalized maximizers `maximizer` of the form argmax_y θᵀg(y) + h(y).
It is compatible with the following layers
- [`PerturbedAdditive`](@ref) (with or without [`FenchelYoungLoss`](@ref))
- [`PerturbedMultiplicative`](@ref) (with or without [`FenchelYoungLoss`](@ref))
- [`SPOPlusLoss`](@ref)
"""
struct GeneralizedMaximizer{F,G,H}
    maximizer::F
    g::G
    h::H
end

GeneralizedMaximizer(f; g=identity_kw, h=zero ∘ eltype_kw) = GeneralizedMaximizer(f, g, h)

function Base.show(io::IO, f::GeneralizedMaximizer)
    (; maximizer, g, h) = f
    return print(io, "GeneralizedMaximizer($maximizer, $g, $h)")
end

# Callable calls the wrapped maximizer
function (f::GeneralizedMaximizer)(θ::AbstractArray{<:Real}; kwargs...)
    return f.maximizer(θ; kwargs...)
end

"""
    objective_value(f, θ, y, kwargs...)

Computes the objective value of given GeneralizedMaximizer `f`, knowing weights `θ` and solution `y`.
"""
function objective_value(f::GeneralizedMaximizer, θ, y; kwargs...)
    return dot(θ, f.g(y; kwargs...)) .+ f.h(y; kwargs...)
end
