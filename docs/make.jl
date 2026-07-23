using Bop
using Documenter

DocMeta.setdocmeta!(Bop, :DocTestSetup, :(using Bop); recursive=true)

makedocs(;
    modules=[Bop],
    authors="AntonOresten <antonoresten@proton.me> and contributors",
    sitename="Bop.jl",
    checkdocs=:public,
    format=Documenter.HTML(;
        canonical="https://docs.jool.space/Bop.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/jool-space/Bop.jl",
    deploy_repo="github.com/jool-space/docs",
    devbranch="main",
    dirname="Bop.jl",
)
