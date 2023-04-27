"""
    PlusIdentity{F}

Naive relaxation of a black-box optimizer where constraints are simply forgotten.

Consider (centering and) normalizing `θ` before applying it.

# Fields
- `maximizer::F`: underlying argmax function

Reference: <https://arxiv.org/abs/2205.15213>
"""
struct PlusIdentity{F}
    maximizer::F
end

function Base.show(io::IO, plusid::PlusIdentity)
    return print(io, "PlusIdentity($(plusid.maximizer)")
end

function (plusid::PlusIdentity)(θ::AbstractArray{<:Real}; kwargs...)
    return plusid.maximizer(θ; kwargs...)
end

function ChainRulesCore.rrule(plusid::PlusIdentity, θ::AbstractArray{<:Real}; kwargs...)
    y = plusid.maximizer(θ; kwargs...)
    plusid_pullback(dy) = NoTangent(), dy
    return y, plusid_pullback
end
