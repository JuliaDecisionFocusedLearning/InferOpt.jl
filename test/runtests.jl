using Test

@testset verbose = true "InferOpt.jl" begin
    @testset "Quality (Aqua.jl)" begin
        include("code/quality.jl")
    end
    
    @testset "Formatting (JuliaFormatter.jl)" begin
        include("code/formatting.jl")
    end
    
    @testset "Correctness (JET.jl)" begin
        include("code/correctness.jl")
    end
    
    @testset "Learning argmax" begin
        include("learning/argmax.jl")
    end
    
    @testset "Learning ranking" begin
        include("learning/ranking.jl")
    end
    
    @testset "Learning paths" begin
        include("learning/paths.jl")
    end
    
    @testset "Learning by imitation" begin
        include("learning/imitation_loss.jl")
    end
end
