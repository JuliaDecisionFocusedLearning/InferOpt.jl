@testitem "Quality (Aqua.jl)" begin
    using Aqua
    Aqua.test_all(InferOpt; ambiguities=false)
end

@testitem "Correctness (JET.jl)" begin
    using JET
    if VERSION >= v"1.9"
        @test_skip JET.test_package(InferOpt; target_defined_modules=true)
    end
end

@testitem "Formatting (JuliaFormatter.jl)" begin
    using JuliaFormatter
    @test format(InferOpt; verbose=false, overwrite=false)
end
