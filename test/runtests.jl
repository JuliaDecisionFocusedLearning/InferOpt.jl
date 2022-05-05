## Imports

using Aqua
using Distributions
using Flux
using InferOpt
using LinearAlgebra
using ProgressMeter
using Random
using Statistics
using Test
using UnicodePlots
using Zygote

## Setup

Random.seed!(63)

SHOW_PLOTS = false

include("utils.jl")

## Tests

@testset verbose = true "InferOpt.jl" begin
    @testset verbose = true "Jacobian approx" begin
        include("jacobian_approx.jl")
    end
    @testset verbose = true "Simple optimizers" begin
        include("simple_optimizers.jl")
    end
    @testset verbose = true "Tutorial" begin
        include("tutorial.jl")
    end
    @testset verbose = true "Code quality (Aqua.jl)" begin
        include("quality.jl")
    end
end
