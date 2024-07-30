struct AdditivePerturbation{F}
    perturbation_dist::F
    ε::Float64
end

"""
θ + εZ
"""
function (pdc::AdditivePerturbation)(θ::AbstractArray)
    (; perturbation_dist, ε) = pdc
    return product_distribution(θ .+ ε * perturbation_dist)
end
