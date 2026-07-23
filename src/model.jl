# The BPE model: vocab + merge ranks. Pieces arrive already byte-level
# mapped, so symbols are plain (multi-)char strings.
struct BPE
    vocab::Dict{String,Int}
    id2tok::Dict{Int,String}
    ranks::Dict{Tuple{String,String},Int}
    ignore_merges::Bool
    cache::Dict{String,Vector{Int}}
end

const CACHE_MAX = 1 << 20

"Tokenize one byte-level piece, appending token ids to `out`."
function bpe!(out::Vector{Int}, m::BPE, piece::String)
    if m.ignore_merges
        id = get(m.vocab, piece, -1)
        if id >= 0
            push!(out, id)
            return out
        end
    end
    cached = get(m.cache, piece, nothing)
    cached !== nothing && return append!(out, cached)

    syms = [string(c) for c in piece]
    while length(syms) > 1
        best, at = typemax(Int), 0
        for i in 1:length(syms)-1
            r = get(m.ranks, (syms[i], syms[i+1]), typemax(Int))
            if r < best
                best, at = r, i
            end
        end
        at == 0 && break
        a, b = syms[at], syms[at+1]
        merged = a * b
        # Merge every (leftmost-first, non-overlapping) occurrence of the
        # selected pair — matches HF's rank-queue result.
        w = sizehint!(String[], length(syms))
        i = 1
        while i <= length(syms)
            if i < length(syms) && syms[i] == a && syms[i+1] == b
                push!(w, merged)
                i += 2
            else
                push!(w, syms[i])
                i += 1
            end
        end
        syms = w
    end

    ids = Vector{Int}(undef, length(syms))
    for (i, s) in pairs(syms)
        id = get(m.vocab, s, -1)
        id < 0 && error("Bop: symbol not in vocab: $(repr(s))")
        ids[i] = id
    end
    length(m.cache) < CACHE_MAX && (m.cache[piece] = ids)
    return append!(out, ids)
end
