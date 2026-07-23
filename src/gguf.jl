# Loading a tokenizer from GGUF metadata. GGUF stores the same data as
# tokenizer.json (tokens, merges, token types) except the pre-tokenizer,
# which it names instead of embedding — `PRE_TOKENIZERS` maps those names
# to their split patterns (the same table llama.cpp hardcodes). Every
# entry must be differentially verified against the model family's
# tokenizer.json before being added.

# ---------------------------------------------------------------------
# Minimal GGUF metadata reader: header + KV section only. Reading stops
# before tensor infos, so a file truncated after the KVs parses fine.
# ---------------------------------------------------------------------

function read_gguf_string(io::IO)
    len = read(io, UInt64)
    return String(read(io, len))
end

function read_gguf_value(io::IO, ty::UInt32)
    ty == 0x8 && return read_gguf_string(io)
    if ty == 0x9
        et = read(io, UInt32)
        n = read(io, UInt64)
        return [read_gguf_value(io, et) for _ in 1:n]
    end
    ty == 0x0 && return read(io, UInt8)
    ty == 0x1 && return read(io, Int8)
    ty == 0x2 && return read(io, UInt16)
    ty == 0x3 && return read(io, Int16)
    ty == 0x4 && return read(io, UInt32)
    ty == 0x5 && return read(io, Int32)
    ty == 0x6 && return read(io, Float32)
    ty == 0x7 && return read(io, UInt8) != 0
    ty == 0xa && return read(io, UInt64)
    ty == 0xb && return read(io, Int64)
    ty == 0xc && return read(io, Float64)
    error("Bop: unknown GGUF value type $ty")
end

"""
    gguf_metadata(path) -> Dict{String,Any}

Read the metadata key-values of a GGUF file. Tensor data is never
touched — reading stops after the KV section, so a file truncated
there parses fine.
"""
function gguf_metadata(path::AbstractString)
    open(path) do io
        read(io, UInt32) == 0x46554747 || error("Bop: not a GGUF file: $path")
        version = read(io, UInt32)
        version in (2, 3) || error("Bop: unsupported GGUF version $version")
        read(io, UInt64) # tensor count
        n_kv = read(io, UInt64)
        md = Dict{String,Any}()
        for _ in 1:n_kv
            k = read_gguf_string(io)
            ty = read(io, UInt32)
            md[k] = read_gguf_value(io, ty)
        end
        return md
    end
end

# ---------------------------------------------------------------------
# The pre-tokenizer name table. Patterns are copied verbatim from each
# family's tokenizer.json and pinned by tests against the paired asset.
# ---------------------------------------------------------------------

"""
    PreSpec(splits, normalizer, ignore_merges, use_regex)

Pre-tokenization recipe for one GGUF `tokenizer.ggml.pre` name:
`splits` is a vector of `(pattern, keep)` Split stages (`keep ∈
(:both, :gaps, :matches)`), `normalizer` an optional Unicode form
(e.g. `:NFC`), and `use_regex` selects the built-in GPT-2 ByteLevel
pattern instead of explicit splits. Fields must be transcribed
verbatim from the family's `tokenizer.json`.
"""
struct PreSpec
    splits::Vector{Tuple{String,Symbol}} # (pattern, keep) per Split stage
    normalizer::Union{Nothing,Symbol}
    ignore_merges::Bool
    use_regex::Bool # ByteLevel builtin GPT-2 pattern instead of Splits
end

"""
    PRE_TOKENIZERS :: Dict{String,PreSpec}

The GGUF pre-tokenizer name table. GGUF stores a *name* (e.g.
`"qwen35"`) where `tokenizer.json` embeds the actual split pattern;
this maps the names llama.cpp's converter emits to their
[`PreSpec`](@ref)s. Names are bound via llama.cpp's tokenizer-checksum
table (hence `"dbrx"` covering Phi-4 and OLMo-2, and `"gpt-4o"`
covering gpt-oss); every entry is generated from the family's
`tokenizer.json` and pinned by tests against the paired asset. To
support a new family, add its entry **and** a differential test
against that family's `tokenizer.json`.

Note: GGUF cannot carry added-token `lstrip`/`rstrip` (Phi-4 sets them
in `tokenizer.json`) — the GGUF path diverges from HF there, as does
llama.cpp itself.
"""
const PRE_TOKENIZERS = Dict{String,PreSpec}(
    "qwen35" => PreSpec(
        [("(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+", :both)],
        :NFC, false, false),
    "qwen2" => PreSpec(
        [("(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+", :both)],
        :NFC, false, false),
    "llama-bpe" => PreSpec(
        [("(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+", :both)],
        nothing, true, false),
    "glm4" => PreSpec(
        [("(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+", :both)],
        nothing, true, false),
    "gpt-4o" => PreSpec(
        [("[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+", :both)],
        nothing, true, false),
    "tekken" => PreSpec(
        [("[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+|[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+", :both)],
        nothing, true, false),
    "deepseek-v3" => PreSpec(
        [("\\p{N}{1,3}", :both),
         ("[一-龥぀-ゟ゠-ヿ]+", :both),
         ("[!\"#\$%&'()*+,\\-./:;<=>?@\\[\\\\\\]^_`{|}~][A-Za-z]+|[^\r\n\\p{L}\\p{P}\\p{S}]?[\\p{L}\\p{M}]+| ?[\\p{P}\\p{S}]+[\r\n]*|\\s*[\r\n]+|\\s+(?!\\S)|\\s+", :both)],
        nothing, false, false),
    "dbrx" => PreSpec(
        [("(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+", :matches)],
        nothing, false, false),
    "modern-bert" => PreSpec(
        [],
        :NFC, false, true),
    "gpt-2" => PreSpec(
        [],
        nothing, false, true),
)
# DeepSeek-V3's hub tokenizer.json changed after llama.cpp recorded its
# checksum; current conversions emit "joyai-llm" (a same-tokenizer clone
# recorded later), older GGUFs say "deepseek-v3". Same spec.
PRE_TOKENIZERS["joyai-llm"] = PRE_TOKENIZERS["deepseek-v3"]

# Token types, per llama.cpp's llama_token_type.
const GGUF_TOKEN_CONTROL = 3
const GGUF_TOKEN_USER_DEFINED = 4

"""
    from_gguf(path_or_metadata) -> Tokenizer

Load a tokenizer directly from GGUF metadata — either a path to a
`.gguf` file or a metadata dict from [`gguf_metadata`](@ref). Requires
`tokenizer.ggml.model == "gpt2"` (byte-level BPE) and a
`tokenizer.ggml.pre` name present in [`PRE_TOKENIZERS`](@ref); anything
else errors loudly.
"""
function from_gguf(md::AbstractDict)
    model = get(md, "tokenizer.ggml.model", nothing)
    model == "gpt2" ||
        error("Bop: unsupported GGUF tokenizer model $(repr(model)) (byte-level BPE \"gpt2\" only)")
    pre = get(md, "tokenizer.ggml.pre", nothing)
    spec = get(PRE_TOKENIZERS, pre, nothing)
    spec === nothing && error(
        "Bop: unknown GGUF pre-tokenizer name $(repr(pre)) — its split pattern " *
        "must be added to Bop.PRE_TOKENIZERS (and verified against the family's tokenizer.json)")

    tokens = md["tokenizer.ggml.tokens"]
    vocab = Dict{String,Int}()
    sizehint!(vocab, length(tokens))
    id2tok = Dict{Int,String}()
    sizehint!(id2tok, length(tokens))
    for (i, tk) in enumerate(tokens)
        s = String(tk)
        vocab[s] = i - 1
        id2tok[i-1] = s
    end
    merges = md["tokenizer.ggml.merges"]
    bpe = build_bpe(vocab, id2tok, merges, spec.ignore_merges)

    added = Dict{String,AddedToken}()
    id2added = Dict{Int,AddedToken}()
    types = get(md, "tokenizer.ggml.token_type", nothing)
    if types !== nothing
        length(types) == length(tokens) || error("Bop: GGUF token_type length mismatch")
        for (i, ty) in enumerate(types)
            (ty == GGUF_TOKEN_CONTROL || ty == GGUF_TOKEN_USER_DEFINED) || continue
            t = AddedToken(i - 1, String(tokens[i]), ty == GGUF_TOKEN_CONTROL, false, false)
            isempty(t.content) && continue
            added[t.content] = t
            id2added[t.id] = t
        end
    end

    splits = [Splitter(Regex(onigify(p)), keep) for (p, keep) in spec.splits]
    normalizer = spec.normalizer === nothing ? nothing :
                 let form = spec.normalizer
        s::String -> Unicode.normalize(s, form)
    end

    # llama.cpp's default for a missing add_bos_token key varies by model
    # lineage; we default to false, which is correct for every family in
    # PRE_TOKENIZERS whose GGUFs were checked (they set the key explicitly).
    # A family relying on an implicit-true default would need its PreSpec
    # to say so — caught by the paired-json differential test when added.
    bos = get(md, "tokenizer.ggml.bos_token_id", nothing)
    template = if Bool(get(md, "tokenizer.ggml.add_bos_token", false)) && bos !== nothing
        [Int(bos), -1]
    else
        nothing
    end

    return Tokenizer(bpe, normalizer, nothing, splits, false, spec.use_regex,
        added, build_added_matcher(added), id2added, template, nothing)
end

from_gguf(path::AbstractString) = from_gguf(gguf_metadata(path))
