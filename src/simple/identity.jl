"""
    IdentityRelaxation <: AbstractOptimizationLayer

Naive relaxation of a black-box optimizer where constraints are simply forgotten.

Consider (centering and) normalizing `θ` before applying it.

# Fields
- `maximizer`: underlying argmax function

Reference: <https://arxiv.org/abs/2205.15213>
"""
struct IdentityRelaxation{F} <: AbstractOptimizationLayer
    maximizer::F
end

function Base.show(io::IO, id::IdentityRelaxation)
    return print(io, "IdentityRelaxation($(id.maximizer)")
end

function (id::IdentityRelaxation)(θ::AbstractArray; kwargs...)
    return id.maximizer(θ; kwargs...)
end

function ChainRulesCore.rrule(id::IdentityRelaxation, θ::AbstractArray; kwargs...)
    y = id.maximizer(θ; kwargs...)
    id_pullback(dy) = NoTangent(), dy
    return y, id_pullback
end
