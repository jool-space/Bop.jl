# Compare Bop against the HF oracle produced by test/scripts/gen_fuzz.py.
using Bop: Bop, encode, decode
using JSON

fuzz = JSON.parse(read(joinpath(@__DIR__, "..", "fuzz.json"), String))

# Bop.decode returns raw bytes; HF runs them through Rust's
# String::from_utf8_lossy. On prefixes of valid UTF-8 (the only invalid
# shapes a truncated encoding can produce) Julia's malformed-Char grouping
# matches Rust's maximal-subpart rule, so one invalid Char ↔ one U+FFFD.
lossy(s::AbstractString) = join(isvalid(c) ? c : '�' for c in s)
total = 0
for (name, cases) in pairs(fuzz)
    tok = Bop.from_file(joinpath(@__DIR__, "..", "assets", String(name), "tokenizer.json"))
    fails = 0
    for c in cases
        enc = encode(tok, String(c.text))
        pids = Vector{Int}(c.ids[1:c.plen])
        ok = enc.ids == c.ids &&
            decode(tok, enc.ids; skip_special_tokens = false) == c.dec &&
            lossy(decode(tok, pids; skip_special_tokens = false)) == c.pdec
        if !ok
            fails += 1
            fails <= 3 && println("MISMATCH [$name]: ", repr(String(c.text)))
        end
    end
    global total += fails
    println(rpad(String(name), 28), fails == 0 ? "$(length(cases))/$(length(cases)) ok" : "($fails FAIL)")
end
println(total == 0 ? "ALL CLEAN" : "TOTAL FAILURES: $total")
exit(total == 0 ? 0 : 1)
