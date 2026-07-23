struct AddedToken
    id::Int
    content::String
    special::Bool
    lstrip::Bool # absorb (drop) whitespace to the left of a match
    rstrip::Bool # … and to the right
end

# Pretokenization = the Split stages from tokenizer.json, then the ByteLevel
# stage (optional prefix space + optional built-in GPT-2 regex + byte map).
struct Splitter
    re::Regex
    keep::Symbol # which pieces survive: :both | :gaps | :matches
end

# HF ByteLevel's built-in pattern (use_regex=true), verbatim from tokenizers;
# compiled through `onigify` (load.jl) like every pattern.
const GPT2_PATTERN = raw"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"

struct Tokenizer
    model::BPE
    normalizer::Union{Nothing,Function}    # String -> String
    splits::Vector{Splitter}
    add_prefix_space::Bool
    bytelevel_regex::Bool                  # ByteLevel(use_regex=true)
    added::Dict{String,AddedToken}
    added_re::Union{Nothing,Regex}
    id2added::Dict{Int,AddedToken}
    template::Union{Nothing,Vector{Int}}   # post-processor single template; -1 = the sequence
end

# `tokens` is materialized on first access — most callers only read `ids`.
# `overrides` records added-token surface forms (whitespace absorbed by
# lstrip/rstrip) where they differ from the canonical content, as HF's
# `.tokens` reports the matched text.
mutable struct Encoding
    const tokenizer::Tokenizer
    const ids::Vector{Int}
    const overrides::Vector{Pair{Int,String}}
    _tokens::Union{Nothing,Vector{String}}
end

Encoding(t, ids::Vector{Int}, overrides = Pair{Int,String}[]) =
    Encoding(t, ids, overrides, nothing)

function Base.getproperty(e::Encoding, name::Symbol)
    if name === :tokens
        tokens = getfield(e, :_tokens)
        if tokens === nothing
            t = getfield(e, :tokenizer)::Tokenizer
            tokens = String[token_string(t, i) for i in getfield(e, :ids)]
            for (i, s) in getfield(e, :overrides)
                tokens[i] = s
            end
            setfield!(e, :_tokens, tokens)
        end
        return tokens
    end
    return getfield(e, name)
end

Base.propertynames(::Encoding) = (:ids, :tokens)
Base.show(io::IO, e::Encoding) = print(io, "Bop.Encoding(", length(e.ids), " tokens)")

Base.show(io::IO, t::Tokenizer) =
    print(io, "Bop.Tokenizer(vocab=$(length(t.model.vocab)), added=$(length(t.added)))")

function token_string(t::Tokenizer, id::Int)
    a = get(t.id2added, id, nothing)
    a !== nothing && return a.content
    s = get(t.model.id2tok, id, nothing)
    s === nothing && error("Bop: unknown token id $id")
    return s
end

"Split one piece by `sp`, appending surviving subpieces to `out`."
function split_piece!(out::Vector{SubString{S}}, sp::Splitter, piece::SubString{S}) where {S}
    pos = firstindex(piece)
    for m in eachmatch(sp.re, piece)
        isempty(m.match) && continue
        if m.offset > pos && sp.keep !== :matches
            push!(out, @views piece[pos:prevind(piece, m.offset)])
        end
        sp.keep !== :gaps && push!(out, m.match)
        pos = m.offset + ncodeunits(m.match)
    end
    pos <= lastindex(piece) && sp.keep !== :matches && push!(out, @views piece[pos:end])
    return out
end

function encode_segment!(ids::Vector{Int}, t::Tokenizer, seg::AbstractString,
    cache::PieceCache = piece_cache(t.model))
    isempty(seg) && return ids
    s = t.normalizer === nothing ? seg : t.normalizer(String(seg))::String
    sub = SubString(s)
    pieces = typeof(sub)[sub]
    next = typeof(sub)[]
    for sp in t.splits
        for p in pieces
            split_piece!(next, sp, p)
        end
        pieces, next = next, empty!(pieces)
    end
    # ByteLevel stage, faithful to HF: per piece, prefix space, then regex.
    for p in pieces
        if t.add_prefix_space && !startswith(p, ' ')
            bytelevel_stage!(ids, t, SubString(" " * p), cache)
        else
            bytelevel_stage!(ids, t, p, cache)
        end
    end
    return ids
end

function bytelevel_stage!(ids::Vector{Int}, t::Tokenizer, q::SubString, cache::PieceCache)
    if t.bytelevel_regex
        for sub in split_piece!(typeof(q)[], GPT2_SPLIT, q)
            bpe!(ids, t.model, sub, cache)
        end
    else
        bpe!(ids, t.model, q, cache)
    end
    return ids
end

function encode(t::Tokenizer, text::AbstractString; add_special_tokens::Bool = true)
    ids = Int[]
    overrides = Pair{Int,String}[]
    cache = piece_cache(t.model)
    if t.added_re === nothing
        encode_segment!(ids, t, text, cache)
    else
        # Added tokens are extracted in one pass over the raw text. Known
        # divergences from HF, both unobserved across the differential
        # corpus: HF matches `normalized: true` added tokens after
        # normalization (moot while normalizers are NFC-or-nothing and
        # added contents are NFC-stable ASCII); and the alternation is
        # O(alternatives) per position where HF uses Aho-Corasick — fine
        # at prompt scale even for Mistral-Nemo's 1000 added tokens.
        s = text
        pos = firstindex(s)
        for m in eachmatch(t.added_re, s)
            if m.offset > pos
                encode_segment!(ids, t, (@views s[pos:prevind(s, m.offset)]), cache)
            end
            # A match may include whitespace absorbed by lstrip/rstrip.
            a = get(t.added, m.match, nothing)
            if a === nothing
                a = t.added[strip(m.match)]
                push!(overrides, (length(ids) + 1) => String(m.match))
            end
            push!(ids, a.id)
            pos = m.offset + ncodeunits(m.match)
        end
        pos <= lastindex(s) && encode_segment!(ids, t, (@views s[pos:end]), cache)
    end
    if add_special_tokens && t.template !== nothing
        final = Int[]
        shift = 0
        for e in t.template
            if e == -1
                shift = length(final)
                append!(final, ids)
            else
                push!(final, e)
            end
        end
        ids = final
        shift != 0 && (overrides = [i + shift => s for (i, s) in overrides])
    end
    return Encoding(t, ids, overrides)
end

"Encode a raw byte buffer (interpreted as UTF-8) without copying it."
encode(t::Tokenizer, bytes::AbstractVector{UInt8}; kw...) = encode(t, StringView(bytes); kw...)

function decode(t::Tokenizer, ids::AbstractVector{<:Integer}; skip_special_tokens::Bool = true)
    io = IOBuffer()
    for id in ids
        a = get(t.id2added, Int(id), nothing)
        if a !== nothing
            a.special && skip_special_tokens && continue
            print(io, a.content)
        else
            write_from_bytelevel!(io, token_string(t, Int(id)))
        end
    end
    return String(take!(io))
end
