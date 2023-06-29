"""
optimizer: θ ⟼ argmax θᵀy - Ω(y)
"""
struct Regularized{O,R}
    Ω::R
    optimizer::O
end

function Base.show(io::IO, regularized::Regularized)
    (; optimizer, Ω) = regularized
    return print(io, "Regularized($optimizer, $Ω)")
end

function (regularized::Regularized)(θ::AbstractArray; kwargs...)
    return regularized.optimizer(θ; kwargs...)
end

function compute_regularization(regularized::Regularized, y::AbstractArray)
    return regularized.Ω(y)
end

# Specific constructors

"""
TODO
"""
function SparseArgmax()
    return Regularized(sparse_argmax_regularization, sparse_argmax)
end

"""
TODO
"""
function SoftArgmax()
    return Regularized(soft_argmax_regularization, soft_argmax)
end

"""
TODO
"""
function RegularizedFrankWolfe(linear_maximizer, Ω, Ω_grad, frank_wolfe_kwargs=NamedTuple())
    # TODO : add a warning if DifferentiableFrankWolfe is not imported ?
    return Regularized(
        Ω, FrankWolfeOptimizer(linear_maximizer, Ω, Ω_grad, frank_wolfe_kwargs)
    )
end
