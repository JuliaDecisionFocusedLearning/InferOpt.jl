function isotonic_l2(y::AbstractVector)
    n = length(y)
    target = [i for i in 1:n] # if block i -> j, then target[i] = j and target[j] = i
    c = ones(n)
    sums = copy(y)
    sol = sums ./ c

    i = 1
    while i <= n
        k = target[i] + 1 # start of next block
        if k == n + 1
            break
        end
        if sol[i] < sol[k]  # continue if B and B+ are correctly ordered
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

function isotonic_l2(s, w)
    return isotonic_l2(s .- w)
end

function ChainRulesCore.rrule(rc::RuleConfig, ::typeof(isotonic_l2), y::AbstractVector)
    ŷ = isotonic_l2(y)

    # TODO: probably can do better (without push! allocations)
    widths = [1]
    for i in eachindex(ŷ)
        if i == length(y)
            break
        end
        if !isapprox(ŷ[i], ŷ[i + 1]; atol=1e-9)
            push!(widths, 0)
        end
        widths[end] += 1
    end

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
