## Graph subtyping

struct AcyclicGridGraph{R<:AbstractFloat} <: AbstractGridGraph{R}
    cell_costs::Matrix{R}
end

function Graphs.ne(g::AcyclicGridGraph)
    h, w = size(g)
    return (
        (h - 1) * (w - 1) * 3 +  # topleft rectangle
        (w - 1) * 1 +  # bottom row
        (h - 1) * 1  # bottom row
    )
end

function Graphs.has_edge(g::AcyclicGridGraph, s::Integer, d::Integer)
    if has_vertex(g, s) && has_vertex(g, d)
        is, js = node_coord(g, s)
        id, jd = node_coord(g, d)
        return (s != d) && (0 <= id - is <= 1) && (0 <= jd - js <= 1)  # 3 neighbors max
    else
        return false
    end
end

function Graphs.outneighbors(g::AcyclicGridGraph, s::Integer)
    h, w = size(g)
    i, j = node_coord(g, s)
    possible_neighbors = (  # listed in ascending index order!
        (i + 1, j + 0),  # bottom
        (i + 0, j + 1),  # right
        (i + 1, j + 1)  # bottom right
    )
    neighbors = (
        node_index(g, id, jd) for
        (id, jd) in possible_neighbors if (1 <= id <= h) && (1 <= jd <= w)
    )
    return neighbors
end

function Graphs.inneighbors(g::AcyclicGridGraph, s::Integer)
    h, w = size(g)
    i, j = node_coord(g, s)
    possible_neighbors = (  # listed in ascending index order!
        (i - 1, j - 1),  # top left
        (i + 0, j - 1),  # left
        (i - 1, j + 0)  # top
    )
    neighbors = (
        node_index(g, id, jd) for
        (id, jd) in possible_neighbors if (1 <= id <= h) && (1 <= jd <= w)
    )
    return neighbors
end

function grid_shortest_paths(g::AcyclicGridGraph, s::Integer)
    return grid_topological_sorting(g, s)
end
