using Revise
using Aqua
using InferOpt
using JET
using JuliaFormatter
using Pkg
using Test
using Zygote

includet("utils/dataset.jl")
includet("utils/error.jl")
includet("utils/perf.jl")
includet("utils/pipeline.jl")

function get_pkg_version(name::AbstractString)
    deps = Pkg.dependencies()
    p = only(x for x in values(deps) if x.name == name)
    return p.version
end

@testset verbose = true "InferOpt.jl" begin
    @testset verbose = true "Code quality (Aqua.jl)" begin
        Aqua.test_all(InferOpt; ambiguities=false)
    end
    @testset verbose = true "Code formatting (JuliaFormatter.jl)" begin
        @test format(InferOpt; verbose=false, overwrite=false)
    end
    @testset verbose = true "Code correctness (JET.jl)" begin
        if get_pkg_version("JET") >= v"0.7.11"
            JET.test_package("InferOpt"; toplevel_logger=nothing)
        else
            @test string(JET.report_package(InferOpt)) == "No errors detected\n"
        end
    end
    @testset verbose = true "Jacobian approx" begin
        include("jacobian_approx.jl")
    end
    @testset verbose = true "Argmax" begin
        include("argmax.jl")
    end
    @testset verbose = true "Ranking" begin
        include("ranking.jl")
    end
    @testset verbose = true "Paths" begin
        include("paths.jl")
    end
    @testset verbose = true "Imitation loss" begin
        include("imitation_loss.jl")
    end
    @testset verbose = true "Frank-Wolfe" begin
        include("frank_wolfe.jl")
    end
    @testset verbose = true "Tutorial" begin
        include("tutorial.jl")
    end
end
