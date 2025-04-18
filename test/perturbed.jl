@testitem "Jacobian approx" begin
    using LinearAlgebra
    using Random
    using Test
    using Zygote

    θ = [3, 5, 4, 2]

    perturbed1 = PerturbedAdditive(one_hot_argmax; ε=1.0, nb_samples=1e4, seed=0)
    perturbed1_big = PerturbedAdditive(one_hot_argmax; ε=1.0, nb_samples=1e6, seed=0)

    perturbed2 = PerturbedMultiplicative(one_hot_argmax; ε=1.0, nb_samples=1e4, seed=0)
    perturbed2_big = PerturbedMultiplicative(one_hot_argmax; ε=1.0, nb_samples=1e6, seed=0)

    @testset "PerturbedAdditive" begin
        # Compute jacobian with reverse mode
        jac1 = Zygote.jacobian(perturbed1, θ)[1]
        jac1_big = Zygote.jacobian(perturbed1_big, θ)[1]
        # Only diagonal should be positive
        @test all(diag(jac1) .>= 0)
        @test all(jac1 - Diagonal(jac1) .<= 0)
        # Order of diagonal coefficients should follow order of θ
        @test sortperm(diag(jac1_big)) == sortperm(θ)
        # No scaling with nb of samples
        @test norm(jac1) ≈ norm(jac1_big) rtol = 5e-2
    end

    @testset "PerturbedMultiplicative" begin
        jac2 = Zygote.jacobian(perturbed2, θ)[1]
        jac2_big = Zygote.jacobian(perturbed2_big, θ)[1]
        @test all(diag(jac2_big) .>= 0)
        @test all(jac2_big - Diagonal(jac2_big) .<= 0)
        @test_broken sortperm(diag(jac2_big)) == sortperm(θ)
        @test norm(jac2) ≈ norm(jac2_big) rtol = 5e-2
    end
end

@testitem "PerturbedOracle vs PerturbedAdditive" default_imports = false begin
    include("InferOptTestUtils/src/InferOptTestUtils.jl")
    using InferOpt, .InferOptTestUtils, Random, Test
    using LinearAlgebra, Zygote, Distributions
    Random.seed!(63)

    ε = 1.0
    p(θ) = MvNormal(θ, ε^2 * I)
    oracle(η) = η

    po = PerturbedOracle(oracle, p; nb_samples=1_000, seed=0)
    pa = PerturbedAdditive(oracle; ε, nb_samples=1_000, seed=0)

    θ = randn(10)
    @test po(θ) ≈ pa(θ) rtol = 0.001
    @test all(isapprox.(jacobian(po, θ), jacobian(pa, θ), rtol=0.001))
end

@testitem "Variance reduction" default_imports = false begin
    include("InferOptTestUtils/src/InferOptTestUtils.jl")
    using InferOpt, .InferOptTestUtils, Random, Test
    using LinearAlgebra, Zygote
    Random.seed!(63)

    ε = 1.0
    oracle(η) = η

    pa = PerturbedAdditive(oracle; ε, nb_samples=100, seed=0, variance_reduction=true)
    pa_no_variance_reduction = PerturbedAdditive(
        oracle; ε, nb_samples=100, seed=0, variance_reduction=false
    )
    pm = PerturbedMultiplicative(oracle; ε, nb_samples=100, seed=0, variance_reduction=true)
    pm_no_variance_reduction = PerturbedMultiplicative(
        oracle; ε, nb_samples=100, seed=0, variance_reduction=false
    )

    n = 10
    θ = randn(10)

    Ja = jacobian(pa_no_variance_reduction, θ)[1]
    Ja_reduced_variance = jacobian(pa, θ)[1]

    Jm = jacobian(pm_no_variance_reduction, θ)[1]
    Jm_reduced_variance = jacobian(pm, θ)[1]

    J_true = Matrix(I, n, n)  # exact jacobian is the identity matrix

    @test normalized_mape(Ja, J_true) > normalized_mape(Ja_reduced_variance, J_true)
    @test normalized_mape(Jm, J_true) > normalized_mape(Jm_reduced_variance, J_true)
end

@testitem "Perturbed - small ε convergence" default_imports = false begin
    include("InferOptTestUtils/src/InferOptTestUtils.jl")
    using InferOpt, .InferOptTestUtils, Random, Test
    using LinearAlgebra, Zygote
    Random.seed!(63)

    ε = 1e-12

    already_differentiable(θ) = 2 ./ exp.(θ) .* θ .^ 2 .+ sum(θ)
    pa = PerturbedAdditive(already_differentiable; ε, nb_samples=1e6, seed=0)
    pm = PerturbedMultiplicative(already_differentiable; ε, nb_samples=1e6, seed=0)

    θ = [1.0, 2.0, 3.0, 4.0, 5.0]

    fz = already_differentiable(θ)
    fa = pa(θ)
    fm = pm(θ)
    @test fz ≈ fa rtol = 0.01
    @test fz ≈ fm rtol = 0.01

    Jz = jacobian(already_differentiable, θ)[1]
    Ja = jacobian(pa, θ)[1]
    Jm = jacobian(pm, θ)[1]
    @test Ja ≈ Jz rtol = 0.01
    @test Jm ≈ Jz rtol = 0.01
end
