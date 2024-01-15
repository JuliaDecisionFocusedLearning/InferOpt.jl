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

tutorial_directory = "examples"
tutorial_files = readdir(tutorial_directory)
tutorial_names = [first(split(file, ".")) for file in tutorial_files]

for name in tutorial_names
    tuto_jl_file = joinpath(dirname(@__DIR__), tutorial_directory, "$name.jl")
    tuto_md_dir = joinpath(@__DIR__, "src")
    Literate.markdown(tuto_jl_file, tuto_md_dir; documenter=true, execute=false)
end

makedocs(;
    modules=[InferOpt],
    authors="Guillaume Dalle, Léo Baty, Louis Bouvier, Axel Parmentier",
    repo="https://github.com/axelparmentier/InferOpt.jl/blob/{commit}{path}#{line}",
    sitename="InferOpt.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://axelparmentier.github.io/InferOpt.jl",
        assets=String[],
        repolink="https://github.com/axelparmentier/InferOpt.jl",
    ),
    pages=[
        "Home" => "index.md",
        "background.md",
        "Examples" => ["basics.md", "tutorial.md", "advanced_applications.md"], # ! hardcoded?
        "Algorithms" => ["optim.md", "losses.md"],
        "API reference" => "api.md",
    ],
)

for file in
    [joinpath(@__DIR__, "src", "index.md"), joinpath(@__DIR__, "src", "tutorial.md")]
    rm(file)
end

deploydocs(; repo="github.com/axelparmentier/InferOpt.jl", devbranch="main")
