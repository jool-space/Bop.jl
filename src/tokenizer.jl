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
    # :both | :gaps | :matches keep those pieces; :merged_prev / :merged_next
    # glue each delimiter onto the preceding / following piece.
    keep::Symbol
end

# HF ByteLevel's built-in pattern (use_regex=true), verbatim from tokenizers;
# compiled through `onigify` (load.jl) like every pattern.
const GPT2_PATTERN = raw"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"

"""
    Tokenizer

A byte-level BPE tokenizer with exact HuggingFace `tokenizers` semantics.
Construct with `Tokenizer(path)` (= [`from_file`](@ref)),
[`from_pretrained`](@ref), or [`from_gguf`](@ref); use with
[`encode`](@ref) and [`decode`](@ref).

Immutable after loading and safe to share across tasks and threads
(the piece cache is task-local).
"""
struct Tokenizer
    model::BPE
    normalizer::Union{Nothing,Function}    # String -> String
    # Metaspace pre-tokenizer: (replacement, prepend scheme). Applied after
    # the normalizer; its optional split lives in `splits` like any other.
    metaspace::Union{Nothing,Tuple{String,Symbol}}
    splits::Vector{Splitter}
    add_prefix_space::Bool
    bytelevel_regex::Bool                  # ByteLevel(use_regex=true)
    added::Dict{String,AddedToken}
    # Added-token matcher: first char → candidates sorted longest-first.
    # (A regex alternation dies at PCRE's pattern-size limit — Gemma has
    # 6415 added tokens — and is O(alternatives) besides.)
    added_by_first::Union{Nothing,Dict{Char,Vector{AddedToken}}}
    id2added::Dict{Int,AddedToken}
    template::Union{Nothing,Vector{Int}}   # post-processor single template; -1 = the sequence
    # Decode chain for sentencepiece-converted files (Replace/ByteFallback/
    # Fuse/Strip steps); `nothing` = the byte-level fast path.
    decoder::Union{Nothing,Vector{Tuple}}
end

# `overrides` records added-token surface forms (whitespace absorbed by
# lstrip/rstrip) where they differ from the canonical content, as HF's
# `.tokens` reports the matched text.
"""
    Encoding

The result of [`encode`](@ref). `enc.ids` holds the token ids (0-based,
matching HF); `enc.tokens` the corresponding token strings, materialized
lazily on first access.
"""
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
    if sp.keep === :merged_prev # delimiters end the piece they follow
        for m in eachmatch(sp.re, piece)
            isempty(m.match) && continue
            stop = m.offset + ncodeunits(m.match)
            push!(out, @views piece[pos:prevind(piece, stop)])
            pos = stop
        end
        pos <= lastindex(piece) && push!(out, @views piece[pos:end])
        return out
    elseif sp.keep === :merged_next # delimiters start the piece they precede
        seg = pos
        for m in eachmatch(sp.re, piece)
            isempty(m.match) && continue
            if m.offset > seg
                push!(out, @views piece[seg:prevind(piece, m.offset)])
            end
            seg = m.offset
        end
        seg <= lastindex(piece) && push!(out, @views piece[seg:end])
        return out
    end
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
    cache::PieceCache = piece_cache(t.model); is_first::Bool = true)
    isempty(seg) && return ids
    s = t.normalizer === nothing ? seg : t.normalizer(String(seg))::String
    if t.metaspace !== nothing
        rep, scheme = t.metaspace
        s2 = replace(s, " " => rep)
        if (scheme === :always || (scheme === :first && is_first)) && !startswith(s2, rep)
            s2 = rep * s2
        end
        s = s2
    end
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

"""
    encode(tokenizer, text; add_special_tokens = true) -> Encoding

Tokenize `text` exactly as HuggingFace `tokenizers` would: added/special
tokens are extracted first, the remaining segments are normalized,
pre-tokenized, and BPE-merged. With `add_special_tokens = true` (the
default) the post-processor template (e.g. a BOS prefix) is applied.

`text` may also be a raw byte buffer (`AbstractVector{UInt8}`,
interpreted as UTF-8), encoded without copying.
"""
function encode(t::Tokenizer, text::AbstractString; add_special_tokens::Bool = true)
    ids = Int[]
    overrides = Pair{Int,String}[]
    cache = piece_cache(t.model)
    if t.added_by_first === nothing
        encode_segment!(ids, t, text, cache; is_first = true)
    else
        # Added tokens are extracted in one leftmost-longest pass over the
        # raw text. (Known divergence from HF, unobserved across the
        # corpus: HF matches `normalized: true` added tokens after
        # normalization — moot while normalizers are NFC-or-nothing and
        # added contents are NFC-stable ASCII.)
        s = text
        seg_start = firstindex(s)
        i = seg_start
        last = lastindex(s)
        while i <= last
            bucket = get(t.added_by_first, s[i], nothing)
            a = nothing
            if bucket !== nothing
                for cand in bucket
                    if startswith((@views s[i:end]), cand.content)
                        a = cand
                        break
                    end
                end
            end
            if a === nothing
                i = nextind(s, i)
                continue
            end
            # lstrip absorbs whitespace before the match (dropped from the
            # preceding segment, kept in the reported surface form).
            surf_start = i
            seg_end = i > seg_start ? prevind(s, i) : 0
            if a.lstrip
                while seg_end >= seg_start && isspace(s[seg_end])
                    surf_start = seg_end
                    seg_end = seg_end > seg_start ? prevind(s, seg_end) : 0
                end
            end
            if seg_end >= seg_start
                encode_segment!(ids, t, (@views s[seg_start:seg_end]), cache;
                    is_first = seg_start == firstindex(s))
            end
            i += ncodeunits(a.content)
            if a.rstrip
                while i <= last && isspace(s[i])
                    i = nextind(s, i)
                end
            end
            surf = @views s[surf_start:prevind(s, i)]
            surf != a.content && push!(overrides, (length(ids) + 1) => String(surf))
            push!(ids, a.id)
            seg_start = i
        end
        seg_start <= last && encode_segment!(ids, t, (@views s[seg_start:end]), cache;
            is_first = seg_start == firstindex(s))
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

encode(t::Tokenizer, bytes::AbstractVector{UInt8}; kw...) = encode(t, StringView(bytes); kw...)

"""
    decode(tokenizer, ids; skip_special_tokens = true) -> String

Map (0-based) token ids back to text. Special tokens are omitted unless
`skip_special_tokens = false`. Unknown ids error.
"""
function decode(t::Tokenizer, ids::AbstractVector{<:Integer}; skip_special_tokens::Bool = true)
    t.decoder === nothing || return decode_chain(t, ids, skip_special_tokens)
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

# The generic decoder pipeline (sentencepiece-converted files), faithful
# to HF: kept token strings flow through the steps, then concatenate.
function decode_chain(t::Tokenizer, ids, skip_special_tokens::Bool)
    toks = String[]
    for id in ids
        a = get(t.id2added, Int(id), nothing)
        if a !== nothing
            a.special && skip_special_tokens && continue
            push!(toks, a.content)
        else
            push!(toks, token_string(t, Int(id)))
        end
    end
    for step in t.decoder::Vector{Tuple}
        kind = step[1]
        if kind === :replace
            toks = String[replace(x, step[2]::String => step[3]::String) for x in toks]
        elseif kind === :bytefallback
            out = String[]
            bytes = UInt8[]
            flushbytes!() = if !isempty(bytes)
                push!(out, isvalid(String, String(copy(bytes))) ? String(copy(bytes)) :
                           repeat("�", length(bytes)))
                empty!(bytes)
            end
            for x in toks
                b = ncodeunits(x) == 6 && startswith(x, "<0x") && endswith(x, ">") ?
                    tryparse(UInt8, x[4:5]; base = 16) : nothing
                if b === nothing
                    flushbytes!()
                    push!(out, x)
                else
                    push!(bytes, b)
                end
            end
            flushbytes!()
            toks = out
        elseif kind === :fuse
            toks = String[join(toks)]
        elseif kind === :strip
            c, nstart, nstop = step[2]::Char, step[3]::Int, step[4]::Int
            toks = map(toks) do x
                chars = collect(x)
                cut0 = 0
                for i in 1:min(nstart, length(chars))
                    chars[i] == c ? (cut0 = i) : break
                end
                cut1 = length(chars) + 1
                for i in 0:min(nstop, length(chars))-1
                    chars[end-i] == c ? (cut1 = length(chars) - i) : break
                end
                String(chars[cut0+1:cut1-1])
            end
        else
            error("Bop: unknown decode step $(kind)")
        end
    end
    return join(toks)
end
