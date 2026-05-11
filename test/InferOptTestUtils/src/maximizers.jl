function shortest_path_maximizer(θ::AbstractMatrix; kwargs...)
    h, w = size(θ)
    dist = fill(Inf, h, w)
    prev_i = zeros(Int, h, w)
    prev_j = zeros(Int, h, w)
    dist[1, 1] = -θ[1, 1]
    for j in 1:w, i in 1:h
        (i == 1 && j == 1) && continue
        for (ni, nj) in ((i - 1, j), (i, j - 1))
            if 1 <= ni && 1 <= nj
                d = dist[ni, nj] - θ[i, j]
                if d < dist[i, j]
                    dist[i, j] = d
                    prev_i[i, j] = ni
                    prev_j[i, j] = nj
                end
            end
        end
    end
    mat = zeros(h, w)
    ci, cj = h, w
    while (ci, cj) != (0, 0)
        mat[ci, cj] = 1.0
        ci, cj = prev_i[ci, cj], prev_j[ci, cj]
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
