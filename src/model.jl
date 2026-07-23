# The BPE model. Merging runs in id space: byte-level chars map to base
# token ids up front (`char_id`), and `pairs` maps a packed (left, right)
# id pair to (rank, merged id) — one integer hash per probe, no string
# concatenation in the merge loop. All tables derive mechanically from the
# tokenizer file; nothing here knows which tokenizer it is.
struct BPE
    vocab::Dict{String,Int}
    id2tok::Dict{Int,String}
    char_id::Vector{Int32} # byte-level char codepoint → base token id
    pairs::Dict{UInt64,Tuple{Int32,Int32}} # packed pair → (rank, merged id)
    ignore_merges::Bool
    cache::Dict{String,Vector{Int32}}
end

const RANK_NONE = typemax(Int32)
const CACHE_MAX = 1 << 20

pair_key(l::Int32, r::Int32) = UInt64(reinterpret(UInt32, l)) << 32 | UInt64(reinterpret(UInt32, r))

@inline function pair_rank(m::BPE, l::Int32, r::Int32)
    return get(m.pairs, pair_key(l, r), (RANK_NONE, Int32(0)))
end

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

    ids = Vector{Int32}(undef, length(piece))
    n = 0
    for c in piece
        id = m.char_id[UInt32(c)+1]
        id < 0 && error("Bop: byte-level char $(repr(c)) has no vocab token")
        ids[n+=1] = id
    end
    # ranks[i] caches (rank, merged) for the pair (ids[i], ids[i+1]).
    ranks = Vector{Tuple{Int32,Int32}}(undef, max(n - 1, 0))
    for i in 1:n-1
        ranks[i] = pair_rank(m, ids[i], ids[i+1])
    end
    while true
        best, at = RANK_NONE, 0
        for (i, (r, _)) in pairs(ranks)
            if r < best
                best, at = r, i
            end
        end
        at == 0 && break
        ids[at] = ranks[at][2]
        deleteat!(ids, at + 1)
        deleteat!(ranks, at)
        at <= length(ranks) && (ranks[at] = pair_rank(m, ids[at], ids[at+1]))
        at > 1 && (ranks[at-1] = pair_rank(m, ids[at-1], ids[at]))
    end
    length(m.cache) < CACHE_MAX && (m.cache[piece] = ids)
    return append!(out, ids)
end
