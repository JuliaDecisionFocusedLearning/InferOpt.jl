function shortest_path_maximizer(θ::AbstractMatrix; kwargs...)
    g = GridGraph(-θ; directions=GridGraphs.QUEEN_ACYCLIC_DIRECTIONS)
    path = grid_topological_sort(g, 1, nv(g))
    y = path_to_matrix(g, path)
    return y
end
