using InferOpt
using Documenter

DocMeta.setdocmeta!(InferOpt, :DocTestSetup, :(using InferOpt); recursive=true)

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
    ],
)

deploydocs(;
    repo="github.com/axelparmentier/InferOpt.jl",
    devbranch="main",
)
