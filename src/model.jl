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
    # Char mode (sentencepiece-converted files, `byte_fallback`/`unk_token`):
    # symbols start as chars looked up in `char_ids`; unknown chars fall back
    # to `<0xXX>` byte tokens (`byte_fb`) or the unk id.
    charmode::Bool
    char_ids::Dict{Char,Int32}
    byte_fb::Vector{Int32} # 256 entries, -1 where the vocab has no <0xXX>
    unk::Int32             # -1 if no unk_token
    fuse_unk::Bool
end

const RANK_NONE = typemax(Int32)
const CACHE_MAX = 1 << 20
const MAX_CACHED_PIECE = 256 # bytes; un-pretokenized segments can be huge

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

    ids = m.charmode ? char_ids_of(m, piece) : byte_ids_of(m, piece)
    merge_ids!(m, ids)
    if length(cache) < CACHE_MAX && ncodeunits(piece) <= MAX_CACHED_PIECE
        cache[piece] = ids
    end
    return append!(out, ids)
end

function byte_ids_of(m::BPE, piece::AbstractString)
    ids = Vector{Int32}(undef, ncodeunits(piece))
    for (i, b) in enumerate(codeunits(piece))
        id = @inbounds m.byte_id[b+1]
        id < 0 && error("Bop: byte 0x$(string(b, base=16)) has no vocab token")
        ids[i] = id
    end
    return ids
end

# Faithful to HF's merge_word: per char, vocab lookup, then <0xXX> byte
# fallback (only if every byte token exists), then unk (fused or not).
function char_ids_of(m::BPE, piece::AbstractString)
    ids = Int32[]
    sizehint!(ids, ncodeunits(piece))
    pending_unk = false
    flush!() = (pending_unk && push!(ids, m.unk); pending_unk = false)
    for c in piece
        id = get(m.char_ids, c, Int32(-1))
        if id >= 0
            flush!()
            push!(ids, id)
            continue
        end
        nb = ncodeunits(c)
        cbuf = codeunits(string(c))
        if all(b -> m.byte_fb[b+1] >= 0, cbuf)
            flush!()
            for b in cbuf
                push!(ids, m.byte_fb[b+1])
            end
        elseif m.unk >= 0
            (pending_unk && !m.fuse_unk) && push!(ids, m.unk)
            pending_unk = true
        end # no unk token: the char is dropped, as HF does
    end
    flush!()
    return ids
end

"Run the BPE merge loop over `ids` in place."
function merge_ids!(m::BPE, ids::Vector{Int32})
    n = length(ids)
    n <= 1 && return ids
    # ranks[i] caches (rank, merged) for the pair (ids[i], ids[i+1]).
    ranks = Vector{Tuple{Int32,Int32}}(undef, n - 1)
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
    return ids
end
