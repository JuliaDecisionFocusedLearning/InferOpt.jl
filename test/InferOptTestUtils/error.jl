function mape(x::AbstractVector, y::AbstractVector)
    return 100 * mean(abs((x[i] - y[i]) / x[i]) for i in eachindex(x))
end

function normalized_mape(x::AbstractVector, y::AbstractVector)
    return mape(x / norm(x), y / norm(y))
end

function normalized_mape(x::AbstractArray, y::AbstractArray)
    return normalized_mape(vec(x), vec(y))
end

function hamming_distance(x::AbstractArray, y::AbstractArray)
    return sum(x[i] != y[i] for i in eachindex(x))
end

function normalized_hamming_distance(x::AbstractArray, y::AbstractArray)
    return mean(x[i] != y[i] for i in eachindex(x))
end
