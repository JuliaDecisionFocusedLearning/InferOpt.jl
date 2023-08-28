function projection_l2(z, w)
    p = sortperm(z)
    return z .- isotonic_l2(z[p], sort(w))[invperm(p)]
end

function ChainRulesCore.rrule(
    rc::RuleConfig, ::typeof(projection_l2), z::AbstractVector, w::AbstractVector
)
    y = projection_l2(z, w)

    p = sortperm(z)
    p_inv = invperm(p)

    pw = sortperm(w)
    pw_inv = invperm(pw)

    _, isotonic_pullback = rrule_via_ad(rc, isotonic_l2, z[p], w[pw])

    function projection_pullback(Δy)
        _, δz, δw = isotonic_pullback(Δy[p])
        return NoTangent(), Δy .- δz[p_inv], -(δw[p_inv])[pw_inv]
    end

    return y, projection_pullback
end

function projection_kl(z, w)
    p = sortperm(z)
    return z .- isotonic_kl(z[p], sort(w))[invperm(p)]
end

function ChainRulesCore.rrule(
    rc::RuleConfig, ::typeof(projection_kl), z::AbstractVector, w::AbstractVector
)
    y = projection_kl(z, w)

    p = sortperm(z)
    p_inv = invperm(p)

    pw = sortperm(w)
    pw_inv = invperm(pw)

    _, isotonic_pullback = rrule_via_ad(rc, isotonic_kl, z[p], w[pw])

    function projection_pullback(Δy)
        _, δz, δw = isotonic_pullback(Δy[p])
        return NoTangent(), Δy .- δz[p_inv], -(δw[p_inv])[pw_inv]
    end

    return y, projection_pullback
end
