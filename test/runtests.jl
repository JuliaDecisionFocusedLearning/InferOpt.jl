using Test

@testset verbose = true "InferOpt.jl" begin
    # @testset verbose = true "Code quality (Aqua.jl)" begin
    #     include("quality.jl")
    # end
    # @testset verbose = true "Jacobian approx" begin
    #     include("jacobian_approx.jl")
    # end
    # @testset verbose = true "Argmax" begin
    #     include("argmax.jl")
    # end
    # @testset verbose = true "Ranking" begin
    #     include("ranking.jl")
    # end
    # @testset verbose = true "Paths" begin
    #     include("paths.jl")
    # end
    # @testset verbose = true "Tutorial" begin
    #     include("tutorial.jl")
    # end
    @testset verbose = true "Argmax" begin
        include("argmax_bis.jl")
    end
end
