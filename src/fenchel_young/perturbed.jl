function compute_F_and_y_samples(
    perturbed::AbstractPerturbed{false}, θ::AbstractArray{<:Real}, Z_samples; kwargs...
)
    F_and_y_samples = [
        fenchel_young_F_and_first_part_of_grad(perturbed, θ, Z; kwargs...) for
        Z in Z_samples
    ]
    return F_and_y_samples
end

function compute_F_and_y_samples(
    perturbed::AbstractPerturbed{true}, θ::AbstractArray{<:Real}, Z_samples; kwargs...
)
    return ThreadsX.map(
        Z -> fenchel_young_F_and_first_part_of_grad(perturbed, θ, Z; kwargs...), Z_samples
    )
end

function fenchel_young_F_and_first_part_of_grad(
    perturbed::AbstractPerturbed, θ::AbstractArray{<:Real}; kwargs...
)
    Z_samples = sample_perturbations(perturbed, θ)
    F_and_y_samples = compute_F_and_y_samples(perturbed, θ, Z_samples; kwargs...)
    return mean(first, F_and_y_samples), mean(last, F_and_y_samples)
end

function fenchel_young_F_and_first_part_of_grad(
    perturbed::PerturbedAdditive,
    θ::AbstractArray{<:Real},
    Z::AbstractArray{<:Real};
    kwargs...,
)
    (; maximizer, ε) = perturbed
    θ_perturbed = θ .+ ε .* Z
    y = maximizer(θ_perturbed; kwargs...)
    F = dot(θ_perturbed, y)
    return F, y
end

function fenchel_young_F_and_first_part_of_grad(
    perturbed::PerturbedMultiplicative,
    θ::AbstractArray{<:Real},
    Z::AbstractArray{<:Real};
    kwargs...,
)
    (; maximizer, ε) = perturbed
    eZ = exp.(ε .* Z .- ε^2)
    θ_perturbed = θ .* eZ
    y = maximizer(θ_perturbed; kwargs...)
    F = dot(θ_perturbed, y)
    y_scaled = y .* eZ
    return F, y_scaled
end
