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

    @testset "bytes input" begin
        tok = Bop.from_file(joinpath(ASSETS, "qwen3.5", "tokenizer.json"))
        fx = JSON3.read(read(joinpath(FIXTURES, "qwen3.5.json"), String))
        for case in fx.cases
            text = String(case.text)
            @test encode(tok, Vector{UInt8}(codeunits(text))).ids == case.ids
        end
    end

    @testset "gguf" begin
        # PRE_TOKENIZERS entries must match the paired tokenizer.json —
        # guards against transcription drift in the name table.
        for (pre, asset) in [("qwen35", "qwen3.5"), ("qwen2", "qwen2.5-0.5b-instruct"),
                             ("llama-bpe", "llama-3.2-1b")]
            j = JSON3.read(read(joinpath(ASSETS, asset, "tokenizer.json"), String))
            items = j.pre_tokenizer.pretokenizers
            patterns = [String(x.pattern.Regex) for x in items if String(x.type) == "Split"]
            @test Bop.PRE_TOKENIZERS[pre].patterns == patterns
        end
        # A GGUF-loaded tokenizer must agree with the paired tokenizer.json
        # (and hence with HF) on every fixture case.
        gg = Bop.from_gguf(joinpath(ASSETS, "qwen3.5", "metadata.gguf"))
        js = Bop.from_file(joinpath(ASSETS, "qwen3.5", "tokenizer.json"))
        fx = JSON3.read(read(joinpath(FIXTURES, "qwen3.5.json"), String))
        for case in fx.cases
            text = String(case.text)
            enc = encode(gg, text)
            @test enc.ids == case.ids
            @test encode(gg, text; add_special_tokens = false).ids == case.ids_plain
            @test decode(gg, enc.ids; skip_special_tokens = false) ==
                  decode(js, enc.ids; skip_special_tokens = false)
        end
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
