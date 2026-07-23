# Differential fuzz generator: random strings drawn from adversarial
# character pools (including the unicode edge-codepoint zoo that caught the
# U+180E PCRE/oniguruma divergence), encoded with HF tokenizers as oracle.
# Writes test/fuzz.json (gitignored); compare with test/scripts/run_fuzz.jl:
#   uv run --with tokenizers python3 test/scripts/gen_fuzz.py [seed] [count]
#   julia --project test/scripts/run_fuzz.jl
import glob
import json
import os
import random
import sys

from tokenizers import Tokenizer

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 31415926
COUNT = int(sys.argv[2]) if len(sys.argv) > 2 else 600

POOLS = [
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'",
    "0123456789",
    " \t\n\r\x0b\x0c",
    # unicode whitespace zoo incl. Ogham space, Mongolian vowel separator,
    # figure/narrow spaces, line/para sep, ideographic space
    "\xa0 ᠎        　",
    # format/invisible: soft hyphen, ZWSP/ZWNJ/ZWJ, word joiner, RTL/LTR
    # marks, ALM, BOM
    "\xad​‌‍⁠‎‏؜﻿",
    # case oddities: Kelvin, long s, dotted/dotless I, eszetts, titlecase
    # digraphs, micro/mu
    "Kſẞ\xdfİıǅǆ\xb5μẛ",
    # Cherokee + Deseret (late-Unicode case pairs), astral letters
    "Ꭰꭰ\U00010400\U00010428\U0001d504\U0001d51e",
    # number zoo: superscripts (No), Roman numeral (Nl), circled,
    # fullwidth (Nd), fractions
    "\xb2\xb3\xb9ⅠⅫ①⑩０９\xbd⁄",
    # marks & combining
    "ָ่́̈\U0001f3fb️",
    "你好日本語한국一",
    "🚀🔥👍",
    ".,;:!?()[]{}#@&*+-=/\\|\"<>",
]


def rand_string(rng: random.Random) -> str:
    out = []
    for _ in range(rng.randint(1, 50)):
        pool = rng.choice(POOLS)
        out.append(rng.choice(pool))
        if rng.random() < 0.4:
            out.append(rng.choice(pool))
    return "".join(out)


def main() -> None:
    rng = random.Random(SEED)
    cases = [rand_string(rng) for _ in range(COUNT)]
    result = {}
    for path in sorted(glob.glob("test/assets/*/tokenizer.json")):
        name = os.path.basename(os.path.dirname(path))
        tok = Tokenizer.from_file(path)
        result[name] = [
            {
                "text": t,
                "ids": tok.encode(t).ids,
                "dec": tok.decode(tok.encode(t).ids, skip_special_tokens=False),
            }
            for t in cases
        ]
    json.dump(result, open("test/fuzz.json", "w"), ensure_ascii=False)
    print(f"seed={SEED} count={COUNT} tokenizers={len(result)}")


if __name__ == "__main__":
    main()
