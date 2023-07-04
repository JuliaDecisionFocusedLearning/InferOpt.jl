@testitem "PerturbedOracle vs PerturbedAdditive" default_imports = false begin
    include("InferOptTestUtils/InferOptTestUtils.jl")
    using InferOpt, .InferOptTestUtils, Random, Test
    using LinearAlgebra, Zygote, Distributions
    Random.seed!(63)

    ε = 1.0
    p(θ) = MvNormal(θ, ε^2 * I)
    oracle(η) = η

    po = PerturbedOracle(p, oracle; nb_samples=1_000, seed=0)
    pa = PerturbedAdditive(oracle; ε, nb_samples=1_000, seed=0)

    θ = randn(10)
    @test po(θ) ≈ pa(θ) rtol = 0.001
    @test all(isapprox.(jacobian(po, θ), jacobian(pa, θ), rtol=0.001))
end

@testitem "Variance reduction" default_imports = false begin
    include("InferOptTestUtils/InferOptTestUtils.jl")
    using InferOpt, .InferOptTestUtils, Random, Test
    using LinearAlgebra, Zygote
    Random.seed!(63)

    ε = 1.0
    oracle(η) = η

    pa = PerturbedAdditive(oracle; ε, nb_samples=100, seed=0)
    pm = PerturbedAdditive(oracle; ε, nb_samples=100, seed=0)

    n = 10
    θ = randn(10)

    Ja = jacobian(pa, θ)[1]
    Ja_reduced_variance = jacobian(x -> pa(x; autodiff_variance_reduction=true), θ)[1]

    Jm = jacobian(pm, θ)[1]
    Jm_reduced_variance = jacobian(x -> pm(x; autodiff_variance_reduction=true), θ)[1]

    J_true = Matrix(I, n, n)  # exact jacobian is the identity matrix

    @test normalized_mape(Ja, J_true) > normalized_mape(Ja_reduced_variance, J_true)
    @test normalized_mape(Jm, J_true) > normalized_mape(Jm_reduced_variance, J_true)
end
