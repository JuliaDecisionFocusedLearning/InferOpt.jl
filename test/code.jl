@testitem "Quality (Aqua.jl)" begin
#! format: noindent
using Aqua
Aqua.test_all(InferOpt; ambiguities=false)
end

@testitem "Correctness (JET.jl)" begin
#! format: noindent
using JET
using Zygote
if VERSION >= v"1.8"
    JET.test_package(InferOpt; toplevel_logger=nothing, mode=:typo)
end
end

@testitem "Formatting (JuliaFormatter.jl)" begin
#! format: noindent
using JuliaFormatter
@test format(InferOpt; verbose=false, overwrite=false)
end
