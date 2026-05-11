function shortest_path_maximizer(θ::AbstractMatrix; kwargs...)
    g = GridGraph(-θ)
    h, w = height(g), width(g)
    n = nv(g)
    dist = fill(Inf, n)
    prev = zeros(Int, n)
    dist[1] = vertex_weight(g, 1)
    for v in 1:(n - 1)
        isinf(dist[v]) && continue
        iv, jv = index_to_coord(g, v)
        for (ni, nj) in ((iv + 1, jv), (iv, jv + 1))
            if 1 <= ni <= h && 1 <= nj <= w
                u = coord_to_index(g, ni, nj)
                d = dist[v] + vertex_weight(g, u)
                if d < dist[u]
                    dist[u] = d
                    prev[u] = v
                end
            end
        end
    end
    mat = zeros(h, w)
    v = n
    while v != 0
        i, j = index_to_coord(g, v)
        mat[i, j] = 1.0
        v = prev[v]
    end
    return mat
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
