using Bop
using JSON3
using Test

const FIXTURES = joinpath(@__DIR__, "fixtures")
const ASSETS = joinpath(@__DIR__, "assets")

include("assets.jl")
ensure_assets(ASSETS)

@testset "Bop.jl" begin
    @testset "bytelevel roundtrip" begin
        s = "héllo 🚀\t\n\0"
        io = IOBuffer()
        Bop.write_from_bytelevel!(io, Bop.to_bytelevel(s))
        @test String(take!(io)) == s
    end

    for fixture in sort(readdir(FIXTURES))
        name = replace(fixture, ".json" => "")
        fx = JSON3.read(read(joinpath(FIXTURES, fixture), String))
        tok = Bop.from_file(joinpath(ASSETS, name, "tokenizer.json"))
        @testset "$name" begin
            for case in fx.cases
                text = String(case.text)
                enc = encode(tok, text)
                @test enc.ids == case.ids
                @test enc.tokens == case.tokens
                @test encode(tok, text; add_special_tokens = false).ids == case.ids_plain
                @test decode(tok, enc.ids) == case.decoded
                @test decode(tok, enc.ids; skip_special_tokens = false) == case.decoded_all
            end
        end
    end
end
