function isotonic_l2(y::AbstractVector)
    n = length(y)
    target = [i for i in 1:n] # if block i -> j, then target[i] = j and target[j] = i
    c = ones(n)
    sums = zeros(n) .+ y
    sol = sums ./ c

    i = 1
    while i <= n
        k = target[i] + 1 # start of next block
        if k == n + 1
            break
        end
        if sol[i] > sol[k]  # continue if B and B+ are correctly ordered
            i = k
            continue
        end

        # merge B with B+
        sums[i] = sums[i] + sums[k]
        c[i] = c[i] + c[k]

        sol[i] = sums[i] / c[i]

        k = target[k] + 1  # start of next block
        target[i] = k - 1
        target[k - 1] = i

        if i > 1
            # check if we now need to merge some blocks before i
            i = target[i - 1]
        end
    end

    # reconstruct solution
    i = 1
    while i <= n
        k = target[i] + 1
        sol[(i + 1):(k - 1)] .= sol[i]
        i = k
    end
    return sol
end

# TODO: handle code duplication
function isotonic_l2_with_sizes(y::AbstractVector)
    n = length(y)
    target = [i for i in 1:n] # if block i -> j, then target[i] = j and target[j] = i
    c = ones(n)
    sums = zeros(n) .+ y
    sol = sums ./ c

    nb_blocks = n  # keep track of number of clock

    i = 1
    while i <= n
        k = target[i] + 1 # start of next block
        if k == n + 1
            break
        end
        if sol[i] > sol[k]  # continue if B and B+ are correctly ordered
            i = k
            continue
        end

        # merge B with B+
        sums[i] = sums[i] + sums[k]
        c[i] = c[i] + c[k]

        sol[i] = sums[i] / c[i]

        k = target[k] + 1  # start of next block
        target[i] = k - 1
        target[k - 1] = i
        nb_blocks -= 1  # one less total blocks

        if i > 1
            # check if we now need to merge some blocks before i
            i = target[i - 1]
        end
    end

    # reconstruct solution
    i = 1
    while i <= n
        k = target[i] + 1
        sol[(i + 1):(k - 1)] .= sol[i]
        i = k
    end

    block_sizes = zeros(nb_blocks)
    current_index = 1
    for i in eachindex(block_sizes)
        width = 1
        value = sol[current_index]
        while current_index <= n && isapprox(value, sol[current_index+1])
            current_index += 1
            width += 1
        end
        value = sol[current_index]
        block_sizes[i] = width
    end

    return sol, block_sizes
end

function isotonic_l2(s, w)
    return isotonic_l2(s .- w)
end

function ChainRulesCore.rrule(::typeof(isotonic_l2), y::AbstractVector)
    ŷ, widths = isotonic_l2_with_sizes(y)

    function isotonic_pullback(Δy)
        res = zeros(length(Δy))
        start = 1
        for width in widths
            slice = start:(start + width - 1)
            res[slice] .= sum(Δy[slice]) / width
            start += width
        end
        return NoTangent(), res
    end

    return ŷ, isotonic_pullback
end
