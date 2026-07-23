# Loading a HF `tokenizer.json`. Anything outside the supported subset
# (byte-level BPE) errors loudly rather than mis-tokenizing silently.

function parse_normalizer(n)
    n === nothing && return nothing
    ty = String(n.type)
    if ty == "Sequence"
        fs = Function[]
        for x in n.normalizers
            f = parse_normalizer(x)
            f === nothing || push!(fs, f)
        end
        isempty(fs) && return nothing
        length(fs) == 1 && return fs[1]
        return s::String -> foldl((acc, f) -> f(acc)::String, fs; init = s)
    elseif ty in ("NFC", "NFD", "NFKC", "NFKD")
        form = Symbol(ty)
        return s::String -> Unicode.normalize(s, form)
    end
    error("Bop: unsupported normalizer type $(repr(ty))")
end

re_escape(s::AbstractString) = replace(s, r"([\\^\$\.\|\?\*\+\(\)\[\]\{\}])" => s"\\\1")

# PCRE2's `\s` includes U+180E (Mongolian vowel separator); Oniguruma — the
# engine HF tokenizers uses, and the behavior we must match — dropped it when
# Unicode 6.3 reclassified it as a format char. Rewrite patterns so `\s`
# excludes it, `\S` includes it, and negated classes containing `\s` re-admit
# it. This is the only class divergence surfaced by differential fuzzing.
function onigify(pat::AbstractString)
    out = IOBuffer()
    class = IOBuffer()
    inclass = false
    negated = false
    class_has_s = false
    i = firstindex(pat)
    while i <= lastindex(pat)
        c = pat[i]
        esc = nothing
        if c == '\\' && i < lastindex(pat)
            i = nextind(pat, i)
            esc = pat[i]
        end
        if !inclass
            if esc === 's'
                print(out, "[^\\S\u180e]")
            elseif esc === 'S'
                print(out, "[\\S\u180e]")
            elseif esc !== nothing
                print(out, '\\', esc)
            elseif c == '['
                inclass = true
                negated = false
                class_has_s = false
                truncate(class, 0)
                print(class, '[')
            else
                print(out, c)
            end
        else
            if esc !== nothing
                esc === 's' && (class_has_s = true)
                esc === 'S' && error("Bop: \\S inside a character class is unsupported")
                print(class, '\\', esc)
            elseif c == '^' && position(class) == 1
                negated = true
                print(class, c)
            elseif c == ']'
                print(class, c)
                body = String(take!(class))
                if class_has_s && negated
                    print(out, "(?:", body, "|\u180e)")
                elseif class_has_s
                    error("Bop: \\s inside a positive character class is unsupported")
                else
                    print(out, body)
                end
                inclass = false
            else
                print(class, c)
            end
        end
        i = nextind(pat, i)
    end
    inclass && error("Bop: unterminated character class in pattern")
    return String(take!(out))
end

function parse_pattern(p)
    haskey(p, :Regex) && return Regex(onigify(String(p.Regex)))
    haskey(p, :String) && return Regex(re_escape(String(p.String)))
    error("Bop: unsupported Split pattern $(p)")
end

const GPT2_SPLIT = Splitter(Regex(onigify(GPT2_PATTERN)), :both)

# Returns (splits, add_prefix_space, bytelevel_regex).
function parse_pretokenizer(pt)
    pt === nothing && return (Splitter[], false, false)
    items = String(pt.type) == "Sequence" ? pt.pretokenizers : [pt]
    splits = Splitter[]
    aps = false
    use_regex = false
    for x in items
        ty = String(x.type)
        if ty == "Split"
            invert = Bool(get(x, :invert, false))
            b = String(x.behavior)
            keep = b == "Isolated" ? :both :
                   b == "Removed" ? (invert ? :matches : :gaps) :
                   error("Bop: Split behavior $(repr(b)) unsupported")
            push!(splits, Splitter(parse_pattern(x.pattern), keep))
        elseif ty == "ByteLevel"
            aps = Bool(get(x, :add_prefix_space, false))
            use_regex = Bool(get(x, :use_regex, true))
        else
            error("Bop: unsupported pre_tokenizer type $(repr(ty))")
        end
    end
    return (splits, aps, use_regex)
end

function parse_model(m)
    ty = String(get(m, :type, "BPE"))
    ty == "BPE" || error("Bop: unsupported model type $(repr(ty)) (byte-level BPE only)")
    Bool(get(m, :byte_fallback, false)) &&
        error("Bop: byte_fallback models (sentencepiece-converted) not yet supported")
    Bool(get(m, :fuse_unk, false)) && error("Bop: fuse_unk unsupported")
    something(get(m, :dropout, nothing), 0.0) == 0.0 || error("Bop: BPE dropout unsupported")
    for k in (:continuing_subword_prefix, :end_of_word_suffix)
        isempty(something(get(m, k, nothing), "")) || error("Bop: $(k) unsupported")
    end

    vocab = Dict{String,Int}()
    sizehint!(vocab, length(m.vocab))
    id2tok = Dict{Int,String}()
    sizehint!(id2tok, length(m.vocab))
    for (k, v) in pairs(m.vocab)
        tok = String(k)
        id = Int(v)
        vocab[tok] = id
        id2tok[id] = tok
    end

    return build_bpe(vocab, id2tok, m.merges, Bool(get(m, :ignore_merges, false)))
end

# Assemble a BPE from vocab + merges. Merge entries may be "a b" strings or
# (a, b) pairs. Shared by the tokenizer.json and GGUF loaders.
function build_bpe(vocab::Dict{String,Int}, id2tok::Dict{Int,String}, merges, ignore_merges::Bool)
    # Byte-level alphabet → base token ids. Vocabs may omit chars for bytes
    # that cannot occur in valid UTF-8 (ModernBERT does); those entries stay
    # -1 and error at encode time if ever hit.
    char_id = fill(Int32(-1), 0x180)
    hits = 0
    for b in 0x00:0xff
        c = BYTE2CHAR[b+1]
        id = get(vocab, string(c), -1)
        id >= 0 && (hits += 1)
        char_id[UInt32(c)+1] = id
    end
    hits > 128 || error("Bop: vocab lacks the byte-level alphabet — not a byte-level BPE tokenizer?")

    pair_tbl = Dict{UInt64,Tuple{Int32,Int32}}()
    sizehint!(pair_tbl, length(merges))
    for (i, merge) in enumerate(merges)
        a, b = if merge isa AbstractString
            parts = split(merge, ' ')
            length(parts) == 2 || error("Bop: malformed merge entry $(repr(merge))")
            String(parts[1]), String(parts[2])
        else
            length(merge) == 2 || error("Bop: malformed merge entry")
            String(merge[1]), String(merge[2])
        end
        la, lb = get(vocab, a, -1), get(vocab, b, -1)
        merged = get(vocab, a * b, -1)
        (la < 0 || lb < 0 || merged < 0) &&
            error("Bop: merge $(repr(a)) + $(repr(b)) refers to tokens missing from vocab")
        get!(pair_tbl, pair_key(Int32(la), Int32(lb)), (Int32(i), Int32(merged)))
    end

    return BPE(vocab, id2tok, char_id, pair_tbl, ignore_merges, Dict{String,Vector{Int32}}())
end

# Extract the single-sequence template from the post-processor, if any.
# -1 marks the sequence slot; other entries are literal token ids.
function parse_template(pp, added::Dict{String,AddedToken}, vocab::Dict{String,Int})
    pp === nothing && return nothing
    ty = String(pp.type)
    if ty == "Sequence"
        for x in pp.processors
            t = parse_template(x, added, vocab)
            t === nothing || return t
        end
        return nothing
    elseif ty == "ByteLevel"
        return nothing
    elseif ty == "TemplateProcessing"
        tmpl = Int[]
        for e in pp.single
            if haskey(e, :Sequence)
                String(e.Sequence.id) == "A" && push!(tmpl, -1)
            elseif haskey(e, :SpecialToken)
                name = String(e.SpecialToken.id)
                st = get(pp.special_tokens, Symbol(name), nothing)
                if st !== nothing
                    append!(tmpl, Int.(st.ids))
                else
                    a = get(added, name, nothing)
                    push!(tmpl, a !== nothing ? a.id : vocab[name])
                end
            end
        end
        return tmpl == [-1] ? nothing : tmpl
    end
    error("Bop: unsupported post_processor type $(repr(ty))")
end

function build_added_re(added::Dict{String,AddedToken})
    isempty(added) && return nothing
    entries = sort!(collect(values(added)); by = a -> length(a.content), rev = true)
    return Regex(join(
        ((a.lstrip ? "\\s*" : "") * re_escape(a.content) * (a.rstrip ? "\\s*" : "")
         for a in entries), "|"))
end

function check_decoder(d)
    d === nothing && return
    ty = String(d.type)
    ty == "ByteLevel" && return
    if ty == "Sequence"
        foreach(check_decoder, d.decoders)
        return
    end
    error("Bop: unsupported decoder type $(repr(ty))")
end

function from_json(j)
    model = parse_model(j.model)
    normalizer = parse_normalizer(get(j, :normalizer, nothing))
    splits, aps, use_regex = parse_pretokenizer(get(j, :pre_tokenizer, nothing))
    check_decoder(get(j, :decoder, nothing))

    added = Dict{String,AddedToken}()
    id2added = Dict{Int,AddedToken}()
    for a in get(j, :added_tokens, ())
        Bool(get(a, :single_word, false)) && error("Bop: added token single_word unsupported")
        t = AddedToken(Int(a.id), String(a.content), Bool(a.special),
            Bool(get(a, :lstrip, false)), Bool(get(a, :rstrip, false)))
        isempty(t.content) && continue
        added[t.content] = t
        id2added[t.id] = t
    end
    added_re = build_added_re(added)

    template = parse_template(get(j, :post_processor, nothing), added, model.vocab)

    return Tokenizer(model, normalizer, splits, aps, use_regex,
        added, added_re, id2added, template)
end

"Load a tokenizer from a HF `tokenizer.json` file."
from_file(path::AbstractString) = from_json(JSON3.read(read(path, String)))

Tokenizer(path::AbstractString) = from_file(path)
