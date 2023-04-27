@testitem "Frank-Wolfe" begin
    using FrankWolfe
    using Random
    using Statistics
    using Test
    using Zygote

    Random.seed!(63)

    d = 100
    x0 = ones(d) / d
    θ = rand(d)
    v = rand(d)
    rc = Zygote.ZygoteRuleConfig()

    _, pullback_sparse_argmax = rrule_via_ad(rc, sparse_argmax, θ)

    fw_kwargs = (max_iteration=500, epsilon=1e-5)

    ## DifferentiableFrankWolfe

    f(x, θ) = half_square_norm(x - θ)
    f_grad1(x, θ) = x - θ
    lmo = FrankWolfe.UnitSimplexOracle(1.0)

    dfw = DifferentiableFrankWolfe(f, f_grad1, lmo)
    _, pullback_dfw = rrule_via_ad(rc, dfw, θ, x0; fw_kwargs=fw_kwargs)

    @testset "DifferentiableFrankWolfe" begin
        @test mean(abs, dfw(θ, x0; fw_kwargs=fw_kwargs) - sparse_argmax(θ)) < 1e-3
        @test mean(abs, pullback_dfw(v)[2] - pullback_sparse_argmax(v)[2]) < 1e-3
    end

    ## RegularizedGeneric

    regularized = RegularizedGeneric(one_hot_argmax, half_square_norm, identity)
    _, pullback_regularized = rrule_via_ad(rc, regularized, θ; fw_kwargs=fw_kwargs)

    @testset "RegularizedGeneric" begin
        @test mean(abs, regularized(θ; fw_kwargs=fw_kwargs) - sparse_argmax(θ)) < 1e-3
        @test mean(abs, pullback_regularized(v)[2] - pullback_sparse_argmax(v)[2]) < 1e-3
    end
end
