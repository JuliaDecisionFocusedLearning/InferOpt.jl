struct SymmetricGridGraph{R<:AbstractFloat} <: AbstractGridGraph{R}
    cell_costs::Matrix{R}
end

function Graphs.ne(g::SymmetricGridGraph)
    h, w = size(g)
    return (
        (h - 2) * (w - 2) * 8 +  # central nodes
        2 * (h - 2) * 5 +  # vertical borders
        2 * (w - 2) * 5 +  # horizontal borders
        2 * 2 * 3  # corners
    )
end

function Graphs.outneighbors(g::SymmetricGridGraph, s::Integer)
    h, w = size(g)
    i, j = node_coord(g, s)
    possible_neighbors = ( # listed in ascending index order!
        (i - 1, j - 1),  # top left
        (i + 0, j - 1),  # left
        (i + 1, j - 1),  # bottom left
        (i - 1, j + 0),  # top
        (i + 1, j + 0),  # bottom
        (i - 1, j + 1),  # top right
        (i + 0, j + 1),  # right
        (i + 1, j + 1),  # bottom right
    )
    neighbors = (
        node_index(g, id, jd) for
        (id, jd) in possible_neighbors if (1 <= id <= h) && (1 <= jd <= w)
    )
    return neighbors
end

Graphs.inneighbors(g::SymmetricGridGraph, d::Integer) = outneighbors(g, d)

function grid_shortest_paths(g::SymmetricGridGraph, s::Integer)
    return grid_dijkstra(g, s)
end
