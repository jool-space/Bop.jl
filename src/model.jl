# The BPE model. Merging runs in id space over raw bytes: `byte_id` maps
# each input byte to its base token id (the byte-level indirection is baked
# into the table at load), and `pairs` maps a packed (left, right) id pair
# to (rank, merged id) — one integer hash per probe, no strings in the
# merge loop. `rawvocab` keys whole-piece lookups (ignore_merges) by raw
# bytes; pieces are probed as zero-copy views. All tables derive
# mechanically from the tokenizer file and are read-only after load, so a
# Tokenizer is safe to share across tasks — the piece-memo cache lives in
# task-local storage (see `piece_cache`), never in the shared struct.
struct BPE
    vocab::Dict{String,Int}
    id2tok::Dict{Int,String}
    rawvocab::Dict{String,Int} # raw bytes → id (byte-level-pure tokens only)
    byte_id::Vector{Int32} # input byte → base token id
    pairs::Dict{UInt64,Tuple{Int32,Int32}} # packed pair → (rank, merged id)
    ignore_merges::Bool
end

const RANK_NONE = typemax(Int32)
const CACHE_MAX = 1 << 20

const PieceCache = Dict{String,Vector{Int32}}

# The piece memo is filled lazily during encoding, so sharing one Dict
# across tasks would race (concurrent insertion can rehash under a
# reader). Instead each task lazily gets its own, keyed by the model
# instance; it is fetched once per `encode` call and threaded through.
function piece_cache(m::BPE)
    tls = task_local_storage()
    return get!(() -> PieceCache(), tls, (:bop_piece_cache, objectid(m)))::PieceCache
end

pair_key(l::Int32, r::Int32) = UInt64(reinterpret(UInt32, l)) << 32 | UInt64(reinterpret(UInt32, r))

@inline function pair_rank(m::BPE, l::Int32, r::Int32)
    return get(m.pairs, pair_key(l, r), (RANK_NONE, Int32(0)))
end

"Tokenize one piece (raw text bytes), appending token ids to `out`."
function bpe!(out::Vector{Int}, m::BPE, piece::AbstractString, cache::PieceCache = piece_cache(m))
    if m.ignore_merges
        id = get(m.rawvocab, piece, -1)
        if id >= 0
            push!(out, id)
            return out
        end
    end
    cached = get(cache, piece, nothing)
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
    length(cache) < CACHE_MAX && (cache[piece] = ids)
    return append!(out, ids)
end
