using FrankWolfe
using InferOpt
using Random
using Statistics
using Test
using Zygote

Random.seed!(63)

d = 100
x0 = ones(d) / d;
θ = rand(d);
v = rand(d);
rc = Zygote.ZygoteRuleConfig()

_, pullback_sparse_argmax = rrule_via_ad(rc, sparse_argmax, θ);

fw_kwargs = (max_iteration=500, epsilon=1e-5)

## DifferentiableFrankWolfe

f(x, θ) = half_square_norm(x - θ)
∇ₓf(x, θ) = x - θ
lmo = FrankWolfe.UnitSimplexOracle(1.0)

dfw = DifferentiableFrankWolfe(f, ∇ₓf, lmo)
_, pullback_dfw = rrule_via_ad(rc, dfw, θ, x0; fw_kwargs=fw_kwargs);

@testset verbose = true "DifferentiableFrankWolfe" begin
    @test mean(abs, dfw(θ, x0; fw_kwargs=fw_kwargs) - sparse_argmax(θ)) < 1e-3
    @test mean(abs, pullback_dfw(v)[2] - pullback_sparse_argmax(v)[2]) < 1e-3
end

## RegularizedGeneric

maximizer(θ) = one_hot_argmax(θ)
Ω(y) = half_square_norm(y)
∇Ω(y) = y

regularized = RegularizedGeneric(maximizer, Ω, ∇Ω)
_, pullback_regularized = rrule_via_ad(rc, regularized, θ; fw_kwargs=fw_kwargs);

@testset verbose = true "RegularizedGeneric" begin
    @test mean(abs, regularized(θ; fw_kwargs=fw_kwargs) - sparse_argmax(θ)) < 1e-3
    @test mean(abs, pullback_regularized(v)[2] - pullback_sparse_argmax(v)[2]) < 1e-3
end
