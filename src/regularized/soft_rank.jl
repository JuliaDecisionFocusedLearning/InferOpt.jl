"""
$TYPEDEF

Fast differentiable ranking regularized layer.
It uses an L2 regularization if `is_l2_regularized` is true, else it uses an entropic (kl) regularization.

As an [`AbstractRegularized`](@ref) layer, it can also be used for supervised learning with
a [`FenchelYoungLoss`](@ref).

Reference: <https://arxiv.org/abs/2002.08871>

# Fields
$TYPEDFIELDS
"""
struct SoftRank{is_l2_regularized} <: AbstractRegularized
    "size of the regularization"
    ε::Float64
    "rank in ascending order if false"
    rev::Bool
end

"""
    SoftRank(; ε::Float64=1.0, rev::Bool=false, is_l2_regularized::Bool=true)

Constructor for [`SoftRank`](@ref).

# Arguments
- `ε::Float64=1.0`: size of the regularization
- `rev::Bool=false`: rank in ascending order if false
- `regularization="l2": used regularization, either "l2" or "kl"
"""
function SoftRank(; ε::Float64=1.0, rev::Bool=false, regularization="l2")
    if regularization == "l2"
        return SoftRank{true}(ε, rev)
    elseif regularization == "kl"
        return SoftRank{false}(ε, rev)
    else
        throw(ArgumentError("Choose eaither l2 or kl regularization"))
    end
end

(l::SoftRank{true})(θ; ε=l.ε, rev=l.rev, kwargs...) = soft_rank_l2(θ; ε, rev)
(l::SoftRank{false})(θ; ε=l.ε, rev=l.rev, kwargs...) = soft_rank_kl(θ; ε, rev)
compute_regularization(l::SoftRank{true}, y) = l.ε * half_square_norm(y)
compute_regularization(l::SoftRank{false}, y) = l.ε * dot(y, log.(y) .- 1)

"""
$TYPEDEF

Fast differentiable sorting optimization layer.
It uses an L2 regularization if `is_l2_regularized` is true, else it uses an entropic (kl) regularization.

Reference <https://arxiv.org/abs/2002.08871>

# Fields
$TYPEDFIELDS
"""
struct SoftSort{is_l2_regularized} <: AbstractOptimizationLayer
    "size of the regularization"
    ε::Float64
    "sort in ascending order if false"
    rev::Bool
end

"""
    SoftSort(; ε::Float64=1.0, rev::Bool=false, is_l2_regularized::Bool=true)

Constructor for [`SoftSort`](@ref).

# Arguments
- `ε::Float64=1.0`: size of the regularization
- `rev::Bool=false`: sort in ascending order if false
- `is_l2_regularized::Bool=true`: use l2 regularization if true, else kl regularization
"""
function SoftSort(; ε::Float64=1.0, rev::Bool=false, is_l2_regularized::Bool=true)
    return SoftSort{is_l2_regularized}(ε, rev)
end

(l::SoftSort{true})(θ; ε=l.ε, rev=l.rev) = soft_sort_l2(θ; ε, rev)
(l::SoftSort{false})(θ; ε=l.ε, rev=l.rev) = soft_sort_kl(θ; ε, rev)

"""
    soft_sort(θ::AbstractVector; ε=1.0, rev::Bool=false, regularization=:l2)

Fast differentiable sort of vector θ.

# Arguments
- `θ`: vector to sort

# Keyword (optional) arguments
- `ε::Float64=1.0`: size of the regularization
- `rev::Bool=false`: sort in ascending order if false
- `regularization=:l2`: use l2 regularization if :l2, and kl regularization if :kl

See also [`soft_sort_l2`](@ref) and [`soft_sort_kl`](@ref).
"""
function soft_sort(θ::AbstractVector; ε=1.0, rev::Bool=false, regularization=:l2)
    if regularization == :l2
        return soft_sort_l2(θ; ε, rev)
    elseif regularization == :kl
        return soft_sort_kl(θ; ε, rev)
    else
        throw(DomainError(regularization, "Argument must either :l2 or :kl"))
    end
end

"""
    soft_rank(θ::AbstractVector; ε=1.0, rev::Bool=false)

Fast differentiable ranking of vector θ.

# Arguments
- `θ`: vector to sort

# Keyword (optional) arguments
- `ε::Float64=1.0`: size of the regularization
- `rev::Bool=false`: sort in ascending order if false
- `regularization=:l2`: use l2 regularization if :l2, and kl regularization if :kl

See also [`soft_rank_l2`](@ref) and [`soft_rank_kl`](@ref).
"""
function soft_rank(θ::AbstractVector; ε=1.0, rev::Bool=false, regularization=:l2)
    if regularization == :l2
        return soft_rank_l2(θ; ε, rev)
    elseif regularization == :kl
        return soft_rank_kl(θ; ε, rev)
    else
        throw(DomainError(regularization, "Argument must either :l2 or :kl"))
    end
end

"""
    soft_sort_l2(θ::AbstractVector; ε=1.0, rev::Bool=false)

Sort vector `θ` with l2 regularization.
"""
function soft_sort_l2(θ::AbstractVector; ε=1.0, rev::Bool=false)
    ρ = length(θ):-1:1
    return rev ? projection_l2(ρ ./ ε, θ) : -projection_l2(ρ ./ ε, -θ)
end

"""
    soft_rank_l2(θ::AbstractVector; ε=1.0, rev::Bool=false)

Rank vector `θ` with l2 regularization.
"""
function soft_rank_l2(θ::AbstractVector; ε=1.0, rev::Bool=false)
    ρ = length(θ):-1:1
    return rev ? projection_l2(-θ ./ ε, ρ) : projection_l2(θ ./ ε, ρ)
end

"""
    soft_sort_kl(θ::AbstractVector; ε=1.0, rev::Bool=false)

Sort vector `θ` with kl regularization.
"""
function soft_sort_kl(θ::AbstractVector; ε=1.0, rev::Bool=false)
    ρ = length(θ):-1:1
    return rev ? projection_kl(ρ ./ ε, θ) : -projection_kl(ρ ./ ε, -θ)
end

"""
    soft_rank_kl(θ::AbstractVector; ε=1.0, rev::Bool=false)

Rank vector `θ` with kl regularization.
"""
function soft_rank_kl(θ::AbstractVector; ε=1.0, rev::Bool=false)
    ρ = length(θ):-1:1
    return rev ? projection_kl(-θ ./ ε, ρ) : projection_kl(θ ./ ε, ρ)
end
