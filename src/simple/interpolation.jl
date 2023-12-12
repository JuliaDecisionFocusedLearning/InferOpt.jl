"""
$TYPEDEF

Piecewise-linear interpolation of a black-box optimizer.

Reference: <https://arxiv.org/abs/1912.02175>

# Fields
$TYPEDFIELDS
"""
struct Interpolation{F} <: AbstractOptimizationLayer
    "underlying argmax function"
    maximizer::F
    "smoothing parameter (smaller = more faithful approximation, larger = more informative gradients)"
    λ::Float64
end

function Base.show(io::IO, interpolation::Interpolation)
    (; maximizer, λ) = interpolation
    return print(io, "Interpolation($maximizer, $λ)")
end

Interpolation(maximizer; λ=1.0) = Interpolation(maximizer, float(λ))

function (interpolation::Interpolation)(θ::AbstractArray; kwargs...)
    return interpolation.maximizer(θ; kwargs...)
end

function ChainRulesCore.rrule(interpolation::Interpolation, θ::AbstractArray; kwargs...)
    (; maximizer, λ) = interpolation
    y = maximizer(θ; kwargs...)
    function interpolation_pullback(dy)
        y_λ = maximizer(θ + λ * dy; kwargs...)
        vjp = (y_λ - y) / λ
        return NoTangent(), vjp
    end
    return y, interpolation_pullback
end
