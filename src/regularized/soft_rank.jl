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
function soft_sort(θ::AbstractVector; ε=1.0, rev::Bool=false)
    ρ = 1:length(θ)
    return rev ? -projection(ρ ./ ε, -θ) : projection(ρ ./ ε, θ)
end

"""
    soft_rank(θ::AbstractVector; ε=1.0)
"""
function soft_rank(θ::AbstractVector; ε=1.0, rev::Bool=false)
    ρ = 1:length(θ)
    return rev ? projection(-θ ./ ε, ρ) : projection(θ ./ ε, ρ)
end
