## Shortest path storage

struct ShortestPathTree{R<:AbstractFloat}
    parents::Vector{Int}
    dists::Vector{R}
end

## Dijkstra

function grid_dijkstra(g::AbstractGridGraph{R}, s::Integer) where {R<:AbstractFloat}
    @assert !has_negative_costs(g)
    dists = fill(typemax(R), nv(g))
    parents = zeros(Int, nv(g))
    Q = PriorityQueue{Int,R}()
    dists[s] = zero(R)
    enqueue!(Q, s, zero(R))
    while !isempty(Q)
        u = dequeue!(Q)
        d_u = dists[u]
        for v in outneighbors(g, u)
            dist_through_u = d_u + get_cost(g, v)
            if dist_through_u < dists[v]
                dists[v] = dist_through_u
                parents[v] = u
                Q[v] = dist_through_u
            end
        end
    end
    return ShortestPathTree(parents, dists)
end

## Topological sorting

function grid_topological_sorting(
    g::AbstractGridGraph{R}, s::Integer
) where {R<:AbstractFloat}
    dists = fill(typemax(R), nv(g))
    parents = zeros(Int, nv(g))
    dists[s] = zero(R)
    for u in s:nv(g)
        for v in outneighbors(g, u)
            c_uv = get_cost(g, v)
            if dists[u] + c_uv < dists[v]
                dists[v] = dists[u] + c_uv
                parents[v] = u
            end
        end
    end
    return ShortestPathTree(parents, dists)
end

## Rebuild path

function grid_shortest_paths(g::AbstractGridGraph, s::Integer)
    error("Not implemented")
end

function grid_shortest_path(g::AbstractGridGraph, s::Integer, d::Integer)
    spt = grid_shortest_paths(g, s)
    parents = spt.parents
    v = d
    path = [v]
    while v != s
        v = parents[v]
        pushfirst!(path, v)
    end
    y = zeros(Bool, height(g), width(g))
    for v in path
        i, j = node_coord(g, v)
        y[i, j] = 1
    end
    return y
end

function grid_shortest_path_cost(g::AbstractGridGraph, s::Integer, d::Integer)
    spt = grid_shortest_paths(g, s)
    return spt.dists[d]
end
