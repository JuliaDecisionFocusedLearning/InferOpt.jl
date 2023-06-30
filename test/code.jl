@testitem "Quality (Aqua.jl)" begin
    using Aqua
    using StatsBase
    Aqua.test_all(InferOpt; ambiguities=false)
    Aqua.test_ambiguities(InferOpt; exclude=[StatsBase.TestStat])
end

@testitem "Correctness (JET.jl)" begin
    using JET
    using DifferentiableFrankWolfe
    if VERSION >= v"1.9"
        @test_skip JET.test_package(InferOpt; target_defined_modules=true)
    end
end

@testitem "Formatting (JuliaFormatter.jl)" begin
    using JuliaFormatter
    @test format(InferOpt; verbose=false, overwrite=false)
end
