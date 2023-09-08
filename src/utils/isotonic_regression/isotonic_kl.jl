function isotonic_kl(s::AbstractVector, w::AbstractVector)
    n = length(s)
    target = [i for i in 1:n] # if block i -> j, then target[i] = j and target[j] = i
    logsumexp_s = zeros(n) .+ s
    logsumexp_w = zeros(n) .+ w
    sol = logsumexp_s .- logsumexp_w

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
        logsumexp_s[i] = logaddexp(logsumexp_s[i], logsumexp_s[k])
        logsumexp_w[i] = logaddexp(logsumexp_w[i], logsumexp_w[k])

        sol[i] = logsumexp_s[i] - logsumexp_w[i]

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

function ChainRulesCore.rrule(
    rc::RuleConfig, ::typeof(isotonic_kl), s::AbstractVector, w::AbstractVector
)
    ŷ = isotonic_kl(s, w)
    # @show ŷ s w

    # TODO: probably can do better (without push! allocations)
    widths = [1]
    for i in eachindex(ŷ)
        if i == length(s)
            break
        end
        if !isapprox(ŷ[i], ŷ[i + 1]; atol=1e-9)
            push!(widths, 0)
        end
        widths[end] += 1
    end

    # ! this is for forward autodiff
    # function isotonic_pullback(Δy)
    #     res_s = zeros(length(Δy))
    #     res_w = zeros(length(Δy))
    #     start = 1
    #     for width in widths
    #         slice = start:(start + width - 1)
    #         @show ŷ widths slice softmax(s[slice]) Δy[slice] softmax(w[slice])
    #         println()
    #         res_s[slice] .= dot(softmax(s[slice]), Δy[slice])
    #         res_w[slice] .= -dot(softmax(w[slice]), Δy[slice])
    #         start += width
    #     end
    #     @show res_w
    #     return NoTangent(), res_s, res_w
    # end
    function isotonic_pullback(Δy)
        res_s = zeros(length(Δy))
        res_w = zeros(length(Δy))
        start = 1
        for width in widths
            slice = start:(start + width - 1)
            # @show ŷ widths slice softmax(s[slice]) Δy[slice] softmax(w[slice])
            println()
            res_s[slice] .= sum(Δy[slice]) .* softmax(s[slice])
            res_w[slice] .= -sum(Δy[slice]) .* softmax(w[slice])
            start += width
        end
        # @show res_w
        return NoTangent(), res_s, res_w
    end

    return ŷ, isotonic_pullback
end
