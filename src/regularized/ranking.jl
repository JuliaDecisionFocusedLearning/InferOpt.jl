function ranking(θ::AbstractVector; rev::Bool=false, kwargs...)
    return invperm(sortperm(θ; rev=rev))
end
