using TestItems

@testitem "Quality (Aqua.jl)" begin
    include("code/quality.jl")
end

@testitem "Formatting (JuliaFormatter.jl)" begin
    include("code/formatting.jl")
end

@testitem "Correctness (JET.jl)" begin
    include("code/correctness.jl")
end

@testitem "Learning argmax" begin
    include("learning/argmax.jl")
end

@testitem "Learning ranking" begin
    include("learning/ranking.jl")
end

@testitem "Learning paths" begin
    include("learning/paths.jl")
end

@testitem "Learning by imitation" begin
    include("learning/imitation_loss.jl")
end
