# The GPT-2 byte-level alphabet: a bijection from bytes to printable chars,
# so BPE can operate on arbitrary bytes as ordinary strings. Bytes that are
# printable (and not space-like) map to themselves; the remaining 68 map to
# U+0100.. in order. E.g. 0x20 (space) ↦ 'Ġ'.
const BYTE2CHAR, CHAR2BYTE = let
    b2c = Vector{Char}(undef, 256)
    c2b = Dict{Char,UInt8}()
    n = 0
    for b in 0x00:0xff
        c = if 0x21 <= b <= 0x7e || 0xa1 <= b <= 0xac || 0xae <= b <= 0xff
            Char(b)
        else
            n += 1
            Char(0xff + n)
        end
        b2c[b+1] = c
        c2b[c] = b
    end
    b2c, c2b
end

"Map a piece of text to its byte-level string (each input byte becomes one Char)."
function to_bytelevel(s::AbstractString)
    io = IOBuffer(sizehint = 2 * ncodeunits(s))
    for b in codeunits(s)
        print(io, @inbounds BYTE2CHAR[b+1])
    end
    String(take!(io))
end

"Write the raw bytes a byte-level token stands for. Chars outside the
alphabet (possible in added-token content) pass through as UTF-8."
function write_from_bytelevel!(io::IO, token::AbstractString)
    for c in token
        b = get(CHAR2BYTE, c, nothing)
        b === nothing ? print(io, c) : write(io, b)
    end
    return io
end
