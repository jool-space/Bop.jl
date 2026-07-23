# The BPE model. Merging runs in id space over raw bytes: `byte_id` maps
# each input byte to its base token id (the byte-level indirection is baked
# into the table at load), and `pairs` maps a packed (left, right) id pair
# to (rank, merged id) — one integer hash per probe, no strings in the
# merge loop. `rawvocab` keys whole-piece lookups (ignore_merges) by raw
# bytes; pieces are probed as zero-copy views and only copied into the
# cache on insertion. All tables derive mechanically from the tokenizer
# file; nothing here knows which tokenizer it is.
struct BPE
    vocab::Dict{String,Int}
    id2tok::Dict{Int,String}
    rawvocab::Dict{String,Int} # raw bytes → id (byte-level-pure tokens only)
    byte_id::Vector{Int32} # input byte → base token id
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

"Tokenize one piece (raw text bytes), appending token ids to `out`."
function bpe!(out::Vector{Int}, m::BPE, piece::AbstractString)
    if m.ignore_merges
        id = get(m.rawvocab, piece, -1)
        if id >= 0
            push!(out, id)
            return out
        end
    end
    cached = get(m.cache, piece, nothing)
    cached !== nothing && return append!(out, cached)

    n = ncodeunits(piece)
    ids = Vector{Int32}(undef, n)
    for (i, b) in enumerate(codeunits(piece))
        id = @inbounds m.byte_id[b+1]
        id < 0 && error("Bop: byte 0x$(string(b, base=16)) has no vocab token")
        ids[i] = id
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
