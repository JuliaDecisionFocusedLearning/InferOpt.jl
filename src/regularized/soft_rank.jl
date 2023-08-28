"""
    SoftRank <: AbstractRegularized

Fast differentiable ranking optimization layer.

As an [`AbstractRegularized`](@ref) layer, it can also be used for supervised learning with
a [`FenchelYoungLoss`](@ref).
"""
struct SoftRank <: AbstractRegularized
    ε::Float64
    rev::Bool
end

function SoftRank(; ε::Float64=1.0, rev::Bool=false)
    return SoftRank(ε, rev)
end

compute_regularization(l::SoftRank, y) = l.ε * half_square_norm(y)
(l::SoftRank)(θ) = soft_rank(θ; ε=l.ε, rev=l.rev)

"""
    SoftSort <: AbstractOptimizationLayer

Fast differentiable sorting optimization layer.
"""
struct SoftSort <: AbstractOptimizationLayer
    ε::Float64
    rev::Bool
end

function SoftSort(; ε::Float64=1.0, rev::Bool=false)
    return SoftSort(ε, rev)
end

(l::SoftSort)(θ) = soft_sort(θ; ε=l.ε, rev=l.rev)

"""
    soft_sort(θ::AbstractVector; ε=1.0, rev=false)

Regularized `sort` function.
"""
function soft_sort(θ::AbstractVector; ε=1.0, rev::Bool=false)
    ρ = collect(1:length(θ))
    return rev ? -projection_l2(ρ ./ ε, -θ) : projection_l2(ρ ./ ε, θ)
end

"""
    soft_rank(θ::AbstractVector; ε=1.0, rev=false)
"""
function soft_rank(θ::AbstractVector; ε=1.0, rev::Bool=false)
    ρ = collect(1:length(θ))
    return rev ? projection_l2(-θ ./ ε, ρ) : projection_l2(θ ./ ε, ρ)
end
