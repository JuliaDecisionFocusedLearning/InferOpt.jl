using InferOpt
using LinearAlgebra
using Random
using Test
using Zygote

Random.seed!(63)

θ = [3, 5, 4, 2]

perturbed1 = PerturbedAdditive(one_hot_argmax; ε=2, nb_samples=1000, seed=0)
perturbed2 = PerturbedMultiplicative(one_hot_argmax; ε=0.5, nb_samples=1000, seed=0)

@testset verbose = true "PerturbedAdditive" begin
    # Compute jacobian with reverse mode
    jac1 = Zygote.jacobian(perturbed1, θ)[1]
    # Only diagonal should be positive
    @test all(diag(jac1) .>= 0)
    @test all(jac1 - Diagonal(jac1) .<= 0)
    # Order of diagonal coefficients should follow order of θ
    @test sortperm(diag(jac1)) == sortperm(θ)
end

@testset verbose = true "PerturbedMultiplicative" begin
    jac2 = Zygote.jacobian(perturbed2, θ)[1]
    @test all(diag(jac2) .>= 0)
    @test all(jac2 - Diagonal(jac2) .<= 0)
    @test_broken sortperm(diag(jac2)) == sortperm(θ)
    # This is broken because the diagonal coefficient for θ₃ = 4 is often larger than the one for θ₂ = 5
    # Maybe because θ₃ has the opportunity to *become* the argmax (and hence switch from 0 to 1), whereas θ₂ already *is* the argmax?
end
