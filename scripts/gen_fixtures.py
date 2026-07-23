# Generate differential-test fixtures: run HF `tokenizers` (the Rust library
# Bop replaces) over a battery of adversarial cases and freeze ids/tokens/
# decodes as JSON. Run from repo root:
#   uv run --with tokenizers python3 scripts/gen_fixtures.py
import json
import glob
import os

from tokenizers import Tokenizer

CASES = [
    "",
    " ",
    "Hello, world!",
    "I've said don't — they'll've been O'Brien's. 'S 'T 'RE 'Ve",
    "  leading spaces",
    "trailing spaces   ",
    "multiple   internal    spaces",
    "tabs\tand\t\ttabs",
    "newlines\nand\n\nmore\r\nCRLF\r",
    " \n \n ",
    "\n",
    "word \nspace-then-newline",
    "1234567890 numbers 007 3.14159 1,000,000",
    "café naïve résumé Zürich",
    "é combining and Å ring",  # NFC normalizer must fold these
    "你好世界，这是测试。日本語のテキストです。한국어 텍스트",
    "مرحبا بالعالم! שלום עולם",
    "\U0001f680\U0001f525 emoji \U0001f44d\U0001f3fd test \U0001f1f8\U0001f1ea ok",
    "def f(x): return x**2  # comment",
    "snake_case camelCase HTTPServer XMLHttpRequest2",
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    "abababababababababab",
    "Mixed: 你好 world 123 \U0001f680 café\n\tdone",
    '<html><body attr="x">&amp;</body></html>',
    "https://example.com/path?query=value&other=123#frag",
    " nbsp and em-space linesep parsep",
    "ẞ ß İ ı ǅ ǆ casing K ſ µ μ",
    " \t᠎",  # Mongolian vowel separator: PCRE-vs-oniguruma \s divergence
    "a᠎b ᠎ 1᠎2 \v᠎\r\n ᠎",
    "Ꭰꭰ\U00010400\U00010428 late-unicode case pairs",
    "² ³ Ⅻ ① ０９ ½ ⁄ number zoo",
    "\U0001d544\U0001d552\U0001d565\U0001d559 \U0001d538\U0001d543\U0001d561\U0001d556\U0001d552 astral letters",
    "Ω≈ç√∫˜µ≤≥÷",
    "word​word zero​width joiner‍⁠nb",
    "The quick brown fox jumps over the lazy dog. Pack my box with five dozen "
    "liquor jugs. How vexingly quick daft zebras jump! Sphinx of black quartz, "
    "judge my vow. Two driven jocks help fax my big quiz. 42 is the answer to "
    "life, the universe, and everything — or so they say.",
]


def cases_for(tok: Tokenizer) -> list[str]:
    cases = list(CASES)
    # Embed this tokenizer's own special tokens mid-text and bare.
    specials = [
        t.content
        for i, t in sorted(tok.get_added_tokens_decoder().items())
        if t.special
    ][:2]
    for s in specials:
        cases.append(s)
        cases.append(f"Start {s} middle{s}end")
        cases.append(f"A \n{s}\t B")  # exercises lstrip/rstrip absorption
    return cases


def main() -> None:
    os.makedirs("test/fixtures", exist_ok=True)
    for path in sorted(glob.glob("test/assets/*/tokenizer.json")):
        name = os.path.basename(os.path.dirname(path))
        tok = Tokenizer.from_file(path)
        out = []
        for text in cases_for(tok):
            enc = tok.encode(text)  # add_special_tokens=True
            plain = tok.encode(text, add_special_tokens=False)
            out.append(
                {
                    "text": text,
                    "ids": enc.ids,
                    "tokens": enc.tokens,
                    "ids_plain": plain.ids,
                    "decoded": tok.decode(enc.ids),  # skip_special_tokens=True
                    "decoded_all": tok.decode(enc.ids, skip_special_tokens=False),
                }
            )
        with open(f"test/fixtures/{name}.json", "w") as f:
            json.dump({"tokenizer": name, "cases": out}, f, ensure_ascii=False)
        print(f"{name}: {len(out)} cases")


if __name__ == "__main__":
    main()
