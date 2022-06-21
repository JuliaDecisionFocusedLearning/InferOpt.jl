using Documenter
using InferOpt
using Literate

DocMeta.setdocmeta!(InferOpt, :DocTestSetup, :(using InferOpt); recursive=true)

# Copy README.md into docs/src/index.md (overwriting)

open(joinpath(@__DIR__, "src", "index.md"), "w") do io
    println(
        io,
        """
        ```@meta
        EditURL = "https://github.com/axelparmentier/InferOpt.jl/blob/main/README.md"
        ```
        """,
    )
    # Write the contents out below the meta bloc
    for line in eachline(joinpath(dirname(@__DIR__), "README.md"))
        println(io, line)
    end
end

# Parse test/tutorial.jl into docs/src/tutorial.md (overwriting)

tuto_jl_file = joinpath(dirname(@__DIR__), "test", "tutorial.jl")
tuto_md_dir = joinpath(@__DIR__, "src")
Literate.markdown(tuto_jl_file, tuto_md_dir; documenter=true, execute=false)

makedocs(;
    modules=[InferOpt],
    authors="Axel Parmentier, Guillaume Dalle, Léo Baty, Louis Bouvier",
    repo="https://github.com/axelparmentier/InferOpt.jl/blob/{commit}{path}#{line}",
    sitename="InferOpt.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://axelparmentier.github.io/InferOpt.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Tutorial" => "tutorial.md",
        "Mathematical background" => "math.md",
        "Algorithms" => "algorithms.md",
        "Implementation" => "implementation.md",
        "API reference" => "api.md",
    ],
)

deploydocs(; repo="github.com/axelparmentier/InferOpt.jl", devbranch="main")
