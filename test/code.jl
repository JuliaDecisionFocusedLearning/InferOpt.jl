@testitem "Quality (Aqua.jl)" begin
    using Aqua
    using StatsBase
    Aqua.test_all(InferOpt; ambiguities=false)
    Aqua.test_ambiguities(InferOpt; exclude=[StatsBase.TestStat])
end

@testitem "Correctness (JET.jl)" default_imports = false begin
    using JET
    using DifferentiableFrankWolfe
    using InferOpt
    if VERSION >= v"1.9"
        JET.test_package(InferOpt; target_modules=[InferOpt])
        # TODO: why does the following line fail even though the method is defined in the extension?
        # JET.test_package(InferOpt, target_defined_modules=true)
    end
end

@testitem "Formatting (JuliaFormatter.jl)" begin
    using JuliaFormatter
    @test format(InferOpt; verbose=false, overwrite=false)
end
