"""
    Interpolation{F}

Piecewise-linear interpolation of a black-box optimizer.

# Fields
- `maximizer::F`: underlying argmax function
- `λ::Float64`: smoothing parameter (smaller = more faithful approximation, larger = more informative gradients)
"""
struct Interpolation{F}
    maximizer::F
    λ::Float64
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