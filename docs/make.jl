using Bop
using Documenter

DocMeta.setdocmeta!(Bop, :DocTestSetup, :(using Bop); recursive=true)

makedocs(;
    modules=[Bop],
    authors="AntonOresten <antonoresten@proton.me> and contributors",
    sitename="Bop.jl",
    format=Documenter.HTML(;
        canonical="https://jool-space.github.io/Bop.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/jool-space/Bop.jl",
    devbranch="main",
)
