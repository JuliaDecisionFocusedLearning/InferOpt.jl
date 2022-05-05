abstract type AbstractGridGraph{R<:AbstractFloat} <: AbstractGraph{Int} end

## Basic accessors

Base.size(g::AbstractGridGraph, args...) = size(g.cell_costs, args...)

Base.eltype(::AbstractGridGraph) = Int
Graphs.edgetype(::AbstractGridGraph) = Edge{Int}

Graphs.is_directed(::AbstractGridGraph) = true
Graphs.is_directed(::Type{<:AbstractGridGraph}) = true

height(g::AbstractGridGraph) = size(g, 1)
width(g::AbstractGridGraph) = size(g, 2)

Graphs.nv(g::AbstractGridGraph) = prod(size(g))
Graphs.vertices(g::AbstractGridGraph) = 1:nv(g)
Graphs.has_vertex(g::AbstractGridGraph, v::Integer) = 1 <= v <= nv(g)

## Indexing translators

function node_index(g::AbstractGridGraph, i::Integer, j::Integer)
    h, w = size(g)
    if (1 <= i <= h) && (1 <= j <= w)
        v = (j - 1) * h + (i - 1) + 1  # enumerate column by column
        return v
    else
        return 0
    end
end

function node_coord(g::AbstractGridGraph, v::Integer)
    if has_vertex(g, v)
        h, w = size(g)
        j = (v - 1) ÷ h + 1
        i = (v - 1) % h + 1
        return i, j
    else
        return (0, 0)
    end
end

## Edges

function Graphs.has_edge(g::AbstractGridGraph, s::Integer, d::Integer)
    if has_vertex(g, s) && has_vertex(g, d)
        is, js = node_coord(g, s)
        id, jd = node_coord(g, d)
        return (s != d) && (abs(is - id) <= 1) && (abs(js - jd) <= 1)  # 8 neighbors max
    else
        return false
    end
end

function Graphs.edges(g::AbstractGridGraph)
    return (Edge(s, d) for s in vertices(g) for d in outneighbors(g, s))
end

## Costs

get_cost(g::AbstractGridGraph, v::Integer) = g.cell_costs[v]
get_cost(g::AbstractGridGraph, i::Integer, j::Integer) = g.cell_costs[i, j]
has_negative_costs(g::AbstractGridGraph) = any(<(0.0), g.cell_costs)
