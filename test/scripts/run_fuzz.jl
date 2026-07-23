# Compare Bop against the HF oracle produced by test/scripts/gen_fuzz.py.
using Bop, JSON

fuzz = JSON.parse(read(joinpath(@__DIR__, "..", "fuzz.json"), String))
total = 0
for (name, cases) in pairs(fuzz)
    tok = Bop.from_file(joinpath(@__DIR__, "..", "assets", String(name), "tokenizer.json"))
    fails = 0
    for c in cases
        enc = encode(tok, String(c.text))
        ok = enc.ids == c.ids && decode(tok, enc.ids; skip_special_tokens = false) == c.dec
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
