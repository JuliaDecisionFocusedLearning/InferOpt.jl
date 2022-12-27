"""
    GeneralizedMaximizer{F,G,H}

Wrapper for generalized maximizers `maximizer` of the form argmax_y θᵀg(y) + h(y).
Can be used with ...
"""
# TODO: find better names for GeneralizedMaximizer, g, and h ?
struct GeneralizedMaximizer{F,G,H}
    maximizer::F
    g::G
    h::H
end

GeneralizedMaximizer(f; g=identity, h=zero) = GeneralizedMaximizer(f, g, h)

# Callable calls the wrapped maximizer
function (f::GeneralizedMaximizer)(θ::AbstractArray{<:Real}; kwargs...)
    return f.maximizer(θ; kwargs...)
end

"""
    objective_value(f, θ, y, kwargs...)

Computes the objective value of given GeneralizedMaximizer `f`, knowing weights `θ` and solution `y`.
"""
# TODO: maybe find a better function name
function objective_value(f::GeneralizedMaximizer, θ, y; kwargs...)
    return dot(θ, f.g(y; kwargs...)) .+ f.h(y; kwargs...)
end
