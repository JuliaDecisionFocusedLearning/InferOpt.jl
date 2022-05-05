one_hot_argmax_approximations = [
    Perturbed(one_hot_argmax; ε=0.3, M=1000)
    PerturbedGeneric(one_hot_argmax; noise_dist=θ -> MultivariateNormal(θ, 0.3), M=1000)
]

for approx in one_hot_argmax_approximations
    θ = [0.1, 0.3, 0.5, 0.4, 0.2]
    # Compute jacobian with reverse mode
    jac = Zygote.jacobian(approx, θ)[1]
    # Only diagonal should be positive
    @test all(diag(jac) .> 0)
    @test all(jac - Diagonal(jac) .<= 0)
    # Order of diagonal coefficients should follow order of θ
    @test sortperm(diag(jac)) == sortperm(θ)
end
