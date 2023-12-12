function shortest_path_maximizer(θ::AbstractMatrix; kwargs...)
    g = GridGraph(-θ; directions=GridGraphs.QUEEN_DIRECTIONS_ACYCLIC)
    path = grid_topological_sort(g, 1, nv(g))
    y = path_to_matrix(g, path)
    return y
end

function max_pricing(θ::AbstractVector; instance::AbstractMatrix)
    @assert length(θ) == size(instance, 1)
    @assert length(θ) == size(instance, 2)
    weights = θ .- instance
    return weights .>= 0
end

g(y; instance, kwargs...) = vec(sum(y; dims=2))
h(y; instance) = -sum(dij * yij for (dij, yij) in zip(instance, y))

identity_kw(x; kwargs...) = identity(x)
mse_kw(x, y; agg=mean, kwargs...) = mse(x, y; agg)
