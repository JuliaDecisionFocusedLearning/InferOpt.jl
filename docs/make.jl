using Documenter
using InferOpt
using Literate

DocMeta.setdocmeta!(InferOpt, :DocTestSetup, :(using InferOpt); recursive=true)

# Copy README.md into docs/src/index.md (overwriting)

<<<<<<< HEAD
open(joinpath(@__DIR__, "src", "index.md"), "w") do io
    println(
        io,
        """
        ```@meta
        EditURL = "https://github.com/JuliaDecisionFocusedLearning/InferOpt.jl/blob/main/README.md"
        ```
        """,
    )
    # Write the contents out below the meta bloc
    for line in eachline(joinpath(dirname(@__DIR__), "README.md"))
        println(io, line)
    end
end
=======
cp(
    joinpath(dirname(@__DIR__), "README.md"),
    joinpath(@__DIR__, "src", "index.md");
    force=true,
)
>>>>>>> main

# Parse test/tutorial.jl into docs/src/tutorial.md (overwriting)

tuto_jl_file = joinpath(dirname(@__DIR__), "examples", "tutorial.jl")
tuto_md_dir = joinpath(@__DIR__, "src")
Literate.markdown(tuto_jl_file, tuto_md_dir; documenter=true, execute=false)

makedocs(;
    modules=[InferOpt],
    authors="Guillaume Dalle, Léo Baty, Louis Bouvier, Axel Parmentier",
    repo="https://github.com/JuliaDecisionFocusedLearning/InferOpt.jl/blob/{commit}{path}#{line}",
    sitename="InferOpt.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://juliadecisionfocusedlearning.github.io/InferOpt.jl",
        assets=String[],
        repolink="https://github.com/JuliaDecisionFocusedLearning/InferOpt.jl",
    ),
    pages=[
        "Home" => "index.md",
        "Background" => "background.md",
        "Examples" => ["tutorial.md", "advanced_applications.md"],
        "Algorithms" => ["optim.md", "losses.md"],
        "API reference" => "api.md",
    ],
)

for file in
    [joinpath(@__DIR__, "src", "index.md"), joinpath(@__DIR__, "src", "tutorial.md")]
    rm(file)
end

deploydocs(; repo="github.com/JuliaDecisionFocusedLearning/InferOpt.jl", devbranch="main")
