using Documenter
using InferOpt
using Literate

DocMeta.setdocmeta!(InferOpt, :DocTestSetup, :(using InferOpt); recursive=true)

# Copy README.md into docs/src/index.md (overwriting)

cp(
    joinpath(dirname(@__DIR__), "README.md"),
    joinpath(@__DIR__, "src", "index.md");
    force=true,
)

# Parse test/tutorial.jl into docs/src/tutorial.md (overwriting)

tuto_jl_file = joinpath(dirname(@__DIR__), "examples", "tutorial.jl")
tuto_md_dir = joinpath(@__DIR__, "src")
Literate.markdown(tuto_jl_file, tuto_md_dir; documenter=true, execute=false)

makedocs(;
    modules=[InferOpt],
    authors="Guillaume Dalle, Léo Baty, Louis Bouvier, Axel Parmentier",
    sitename="InferOpt.jl",
    format=Documenter.HTML(),
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
