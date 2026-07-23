using Bop: Bop, encode, decode
using JSON
using Test

using Republic

const FIXTURES = joinpath(@__DIR__, "fixtures")
const ASSETS = joinpath(@__DIR__, "assets")

include("assets.jl")
ensure_assets(ASSETS)

@testset "Bop.jl" begin
    @testset "public API" begin
        for name in (:Tokenizer, :Encoding, :encode, :decode, :encode_batch,
            :decode_batch, :from_file, :from_json, :from_pretrained,
            :from_gguf, :gguf_metadata, :PRE_TOKENIZERS, :PreSpec)
            @test Republic.ispublic(Bop, name)
            @test Base.Docs.hasdoc(Bop, name)
        end
    end

    @testset "bytelevel roundtrip" begin
        s = "héllo 🚀\t\n\0"
        io = IOBuffer()
        Bop.write_from_bytelevel!(io, Bop.to_bytelevel(s))
        @test String(take!(io)) == s
    end

    @testset "inference" begin
        # @inferred checks return-type inference at the API boundary (it
        # does not prove the absence of internal dynamic dispatch).
        tok = Bop.from_file(joinpath(ASSETS, "qwen3.5", "tokenizer.json"))   # NFC + Split
        gp = Bop.from_file(joinpath(ASSETS, "gpt2", "tokenizer.json"))       # builtin regex path
        enc = @inferred encode(tok, "hello world")
        @test enc isa Bop.Encoding
        @test @inferred(encode(tok, Vector{UInt8}(codeunits("hello world")))) isa Bop.Encoding
        @test @inferred(encode(gp, "hello world")) isa Bop.Encoding
        @test @inferred(decode(tok, enc.ids)) isa String
        @test @inferred((e -> e.ids)(enc)) isa Vector{Int}
        @test @inferred((e -> e.tokens)(enc)) isa Vector{String}
        @test @inferred(Bop.bpe!(Int[], tok.model, SubString("abc"))) isa Vector{Int}
        @test @inferred(Bop.split_piece!(SubString{String}[], tok.splits[1], SubString("a b"))) isa Vector{SubString{String}}
        @test @inferred(Bop.encode_segment!(Int[], tok, SubString("hello"))) isa Vector{Int}
        sv = Bop.StringViews.StringView(codeunits("hello"))
        @test @inferred(Bop.encode_segment!(Int[], tok, SubString(sv))) isa Vector{Int}
        @test @inferred(Bop.encode_segment!(Int[], gp, SubString(sv))) isa Vector{Int}
        @test @inferred(Bop.token_string(tok, 100)) isa String
    end

    @testset "bytes input" begin
        tok = Bop.from_file(joinpath(ASSETS, "qwen3.5", "tokenizer.json"))
        fx = JSON.parse(read(joinpath(FIXTURES, "qwen3.5.json"), String))
        for case in fx.cases
            text = String(case.text)
            @test encode(tok, Vector{UInt8}(codeunits(text))).ids == case.ids
        end
    end

    @testset "shared-instance concurrency" begin
        # One Tokenizer shared across tasks: tables are read-only, the
        # piece memo is task-local. Determinism check under contention
        # (a data race here would corrupt caches and scramble ids).
        tok = Bop.from_file(joinpath(ASSETS, "qwen3.5", "tokenizer.json"))
        fx = JSON.parse(read(joinpath(FIXTURES, "qwen3.5.json"), String))
        texts = repeat([String(c.text) for c in fx.cases], 40)
        want = [encode(tok, t).ids for t in texts]
        tasks = [Threads.@spawn [encode(tok, t).ids for t in texts] for _ in 1:8]
        @test all(fetch(task) == want for task in tasks)
    end

    @testset "batch" begin
        tok = Bop.from_file(joinpath(ASSETS, "qwen3.5", "tokenizer.json"))
        texts = ["hello", "world <|im_end|>", ""]
        encs = Bop.encode_batch(tok, texts)
        @test [e.ids for e in encs] == [encode(tok, t).ids for t in texts]
        @test Bop.decode_batch(tok, [e.ids for e in encs]; skip_special_tokens = false) == texts
    end

    @testset "gguf" begin
        # Every PRE_TOKENIZERS entry must match its paired tokenizer.json —
        # guards against transcription drift in the name table. Names come
        # from llama.cpp's converter checksum table ("dbrx" really is what
        # Phi-4 and OLMo-2 GGUFs carry).
        for (pre, asset) in [
            ("qwen35", "qwen3.5"),
            ("qwen2", "qwen2.5-0.5b-instruct"), ("qwen2", "qwen3-0.6b"),
            ("llama-bpe", "llama-3.2-1b"),
            ("glm4", "glm-4.5-air"),
            ("gpt-4o", "gpt-oss-20b"),
            ("tekken", "mistral-nemo-instruct-2407"),
            ("deepseek-v3", "deepseek-v3"), ("joyai-llm", "deepseek-v3"),
            ("dbrx", "phi-4"), ("dbrx", "olmo-2-1124-7b"),
            ("modern-bert", "modernbert-base"),
            ("gpt-2", "gpt2"),
        ]
            j = JSON.parse(read(joinpath(ASSETS, asset, "tokenizer.json"), String))
            pt = j.pre_tokenizer
            items = String(pt.type) == "Sequence" ? pt.pretokenizers : [pt]
            splits = Tuple{String,Symbol}[]
            use_regex = false
            for x in items
                if String(x.type) == "Split"
                    keep = String(x.behavior) == "Isolated" ? :both :
                           x.invert ? :matches : :gaps
                    push!(splits, (String(x.pattern.Regex), keep))
                else
                    use_regex = Bool(get(x, "use_regex", true))
                end
            end
            nm = get(j, "normalizer", nothing)
            norm = nm === nothing ? nothing :
                   String(nm.type) == "Sequence" && isempty(nm.normalizers) ? nothing :
                   Symbol(String(nm.type))
            spec = Bop.PRE_TOKENIZERS[pre]
            @test spec.splits == splits
            @test spec.use_regex == use_regex
            @test spec.normalizer == norm
            @test spec.ignore_merges == Bool(something(get(j.model, "ignore_merges", false), false))
        end
        # A GGUF-loaded tokenizer must agree with the paired tokenizer.json
        # (and hence with HF) on every fixture case.
        gg = Bop.from_gguf(joinpath(ASSETS, "qwen3.5", "metadata.gguf"))
        js = Bop.from_file(joinpath(ASSETS, "qwen3.5", "tokenizer.json"))
        fx = JSON.parse(read(joinpath(FIXTURES, "qwen3.5.json"), String))
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
        fx = JSON.parse(read(joinpath(FIXTURES, fixture), String))
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
