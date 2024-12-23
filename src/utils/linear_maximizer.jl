"""
$TYPEDEF

Wrapper for generic minear maximizers of the form argmax_y θᵀg(y) + h(y).
It is compatible with the following layers
- [`PerturbedAdditive`](@ref) (with or without a [`FenchelYoungLoss`](@ref))
- [`PerturbedMultiplicative`](@ref) (with or without a [`FenchelYoungLoss`](@ref))
- [`SPOPlusLoss`](@ref)

# Fields
$TYPEDFIELDS
"""
@kwdef struct LinearMaximizer{F,G,H}
    "function θ ⟼ argmax_y θᵀg(y) + h(y)"
    maximizer::F
    "function g(y) used in the objective"
    g::G = identity_kw
    "function h(y) used in the objective"
    h::H = zero ∘ eltype_kw
end

function Base.show(io::IO, f::LinearMaximizer)
    (; maximizer, g, h) = f
    return print(io, "LinearMaximizer($maximizer, $g, $h)")
end

"""
$TYPEDSIGNATURES

Constructor for [`LinearMaximizer`](@ref).
"""
function LinearMaximizer(maximizer; g=identity_kw, h=zero ∘ eltype_kw)
    return LinearMaximizer(maximizer, g, h)
end

"""
$TYPEDSIGNATURES

Calls the wrapped maximizer.
"""
function (f::LinearMaximizer)(θ::AbstractArray; kwargs...)
    return f.maximizer(θ; kwargs...)
end

# default is oracles of the form argmax_y θᵀy
objective_value(::Any, θ, y; kwargs...) = dot(θ, y)
apply_g(::Any, y; kwargs...) = y
# apply_h(::Any, y; kwargs...) = zero(eltype(y)) is not needed

"""
$TYPEDSIGNATURES

Computes the objective value of given LinearMaximizer `f`, knowing weights `θ` and solution `y`.
i.e. θᵀg(y) + h(y)
"""
function objective_value(f::LinearMaximizer, θ, y; kwargs...)
    return dot(θ, f.g(y; kwargs...)) .+ f.h(y; kwargs...)
end

"""
$TYPEDSIGNATURES

Applies the function `g` of the LinearMaximizer `f` to `y`.
"""
function apply_g(f::LinearMaximizer, y; kwargs...)
    return f.g(y; kwargs...)
end

# """
# $TYPEDSIGNATURES

# Applies the function `h` of the LinearMaximizer `f` to `y`.
# """
# function apply_h(f::LinearMaximizer, y; kwargs...)
#     return f.h(y; kwargs...)
# end
