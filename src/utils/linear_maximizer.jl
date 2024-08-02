"""
    LinearMaximizer{F,G,H}

Wrapper for generalized maximizers `maximizer` of the form argmax_y θᵀg(y) + h(y).
It is compatible with the following layers
- [`PerturbedAdditive`](@ref) (with or without [`FenchelYoungLoss`](@ref))
- [`PerturbedMultiplicative`](@ref) (with or without [`FenchelYoungLoss`](@ref))
- [`SPOPlusLoss`](@ref)
"""
@kwdef struct LinearMaximizer{G,H,F}
    maximizer::F
    g::G = nothing
    h::H = nothing
end

function Base.show(io::IO, f::LinearMaximizer)
    (; maximizer, g, h) = f
    return print(io, "LinearMaximizer($maximizer, $g, $h)")
end

# Callable calls the wrapped maximizer
function (f::LinearMaximizer)(θ::AbstractArray; kwargs...)
    return f.maximizer(θ; kwargs...)
end

objective_value(::LinearMaximizer{Nothing,Nothing}, θ, y; kwargs...) = dot(θ, y)

"""
    objective_value(f, θ, y, kwargs...)

Computes the objective value of given LinearMaximizer `f`, knowing weights `θ` and solution `y`.
"""
function objective_value(f::LinearMaximizer, θ, y; kwargs...)
    return dot(θ, f.g(y; kwargs...)) .+ f.h(y; kwargs...)
end

apply_g(::LinearMaximizer{Nothing,Nothing}, y; kwargs...) = y
apply_h(::LinearMaximizer{Nothing,Nothing}, y; kwargs...) = zero(eltype(y))

function apply_g(f::LinearMaximizer, y; kwargs...)
    return f.g(y; kwargs...)
end

function apply_h(f::LinearMaximizer, y; kwargs...)
    return f.h(y; kwargs...)
end
