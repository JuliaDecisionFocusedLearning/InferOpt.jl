"""
maximizer corresponds to argmax_y θᵀg(y) + h(y)
"""
struct GeneralizedMaximizer{F,G,H}
    maximizer::F
    g::G
    h::H
end

GeneralizedMaximizer(f; g=identity, h=identity) = GeneralizedMaximizer(f, g, h)

"""
"""
function (f::GeneralizedMaximizer)(θ::AbstractArray{<:Real}; kwargs...)
    return f.maximizer(θ; kwargs...)
end

"""
"""
# TODO: maybe find a better function name
function objective_value(f::GeneralizedMaximizer, θ, y; kwargs...)
    return dot(θ, f.g(y; kwargs...)) .+ f.h(y; kwargs...)
end
