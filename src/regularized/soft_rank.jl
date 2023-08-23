"""
TODO
"""
struct SoftRank <: AbstractRegularized
    ε::Float64
end

"""
TODO
"""
struct SoftSort <: AbstractOptimizationLayer
    ε::Float64
end

(l::SoftRank)(θ) = soft_rank(θ; ε=l.ε)
(l::SoftSort)(θ) = soft_sort(θ; ε=l.ε)

compute_regularization(l::SoftRank, y) = l.ε * half_square_norm(y)

"""
    soft_sort(θ::AbstractVector; ε=1.0)
"""
function soft_sort(θ::AbstractVector; ε=1.0)
    n = length(θ)
    ρ = n:-1:1
    return projection(ρ ./ ε, θ)
end

"""
    soft_rank(θ::AbstractVector; ε=1.0)
"""
function soft_rank(θ::AbstractVector; ε=1.0)
    n = length(θ)
    ρ = n:-1:1
    return projection(-θ ./ ε, ρ)
end
