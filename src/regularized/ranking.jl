function ranking(θ::AbstractVector{<:Real}; rev::Bool=false, kwargs...)
    return invperm(sortperm(θ; rev=rev))
end
