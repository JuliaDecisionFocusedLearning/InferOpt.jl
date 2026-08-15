@testset "Tutorials" begin
    @testitem "Flux Tutorial" begin
        include(joinpath(dirname(@__DIR__), "examples", "tutorial_flux.jl"))
    end

    @testitem "Lux Tutorial" begin
        include(joinpath(dirname(@__DIR__), "examples", "tutorial_lux.jl"))
    end
end
