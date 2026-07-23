# Bop

[![Build Status](https://github.com/jool-space/Bop.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/jool-space/Bop.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/jool-space/Bop.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/jool-space/Bop.jl)

Tokenize text for language models in pure Julia. Bop reads the same
`tokenizer.json` files that HuggingFace models ship — or the metadata
inside a GGUF file — and produces exactly the same token ids, with no
Python, Rust, or conda environment in sight.

```julia
using Bop

tok = Tokenizer("tokenizer.json")       # or Bop.from_pretrained("Qwen/Qwen3-0.6B")
enc = encode(tok, "Hello, world!")
enc.ids                                 # 0-based ids, exactly as HF
enc.tokens                              # token strings, computed on demand
decode(tok, enc.ids)                    # "Hello, world!"
```

This covers the tokenizers used by essentially every current open-weights
LM: the GPT-2/GPT-4 lineage, Qwen, Llama 2/3/4, DeepSeek, Mistral, GLM,
gpt-oss, Phi-3/4, OLMo, SmolLM, and friends — byte-level BPE *and*
sentencepiece-converted BPE (Gemma, TinyLlama, Zephyr: `byte_fallback`,
`Metaspace`, the works). What remains out of scope — Unigram (T5) and
WordPiece (BERT) models — fails loudly at load rather than
mis-tokenizing.

Some things that come for free:

- **GGUF**: `Bop.from_gguf("model.gguf")` builds the tokenizer straight
  from GGUF metadata — no `tokenizer.json` sidecar needed. Eleven
  pre-tokenizer families are covered, each verified against its HF
  counterpart.
- **Threads**: encoding is plain Julia, so it parallelizes with
  `Threads.@spawn` — no GIL, and a single `Tokenizer` is safe to share
  across tasks (its tables are read-only; the piece cache is task-local).
- **Raw bytes**: `encode` accepts `AbstractVector{UInt8}` (an mmap'd
  corpus, say) without copying.
- **Batches**: `Bop.encode_batch` / `Bop.decode_batch`, and
  `encode(tok, text; add_special_tokens = false)` /
  `decode(tok, ids; skip_special_tokens = false)` behave as in HF.

## Why trust it

Matching HF exactly is the whole point, so Bop is tested differentially:
every release is checked against the Rust `tokenizers` library over twelve
real tokenizers (GPT-2, Qwen 2.5/3/3.5, Llama 3.2, DeepSeek-V3, gpt-oss,
Phi-3/Phi-4, Mistral-v0.2/Nemo, GLM-4.5, OLMo-2, ModernBERT, SmolLM2,
TinyLlama, Gemma-3 — seventeen in all) on a battery of
adversarial inputs — unicode whitespace exotica, astral-plane letters,
contraction casing, special tokens glued to whitespace — plus thousands of
randomized fuzz cases. Ids, token strings, and both decode modes match
exactly. Where the regex engines beneath the two implementations genuinely
disagree (PCRE2 still counts U+180E as whitespace; Oniguruma stopped in
Unicode 6.3), Bop rewrites patterns at load so HF behavior wins.

Fixtures regenerate with `uv run --with tokenizers python3
test/scripts/gen_fixtures.py`; the fuzz harness is `test/scripts/gen_fuzz.py` +
`test/scripts/run_fuzz.jl`.

## Speed

Fast enough that tokenization stops being a consideration: a chat prompt
encodes in ~3.5 µs, and bulk text runs at ~12.5 MB/s single-threaded
(about 4× the HF Python stack) or ~55 MB/s on 8 threads sharing one
`Tokenizer` — faster than HF's fully parallel `encode_batch`. Loading a 13 MB tokenizer.json takes
~0.3 s.

## Fine print

Supported `tokenizer.json` components: BPE models (incl. `ignore_merges`,
`byte_fallback`, `fuse_unk`/`unk_token`, both merges formats), NFC-family
/ `Prepend` / `Replace` normalizers, chained `Split` pre-tokenizers (all
four behaviors, `invert`), `Digits`, `Metaspace`, `ByteLevel` (incl.
`add_prefix_space` and the built-in GPT-2 pattern), single-sequence
`TemplateProcessing`, added/special tokens incl. `lstrip`/`rstrip`, and
`ByteLevel` or `Replace`/`ByteFallback`/`Fuse`/`Strip` decoder chains. Offsets and word ids are deliberately out of
scope. One GGUF caveat: the format cannot carry added-token
`lstrip`/`rstrip` flags (Phi-4 uses them), so GGUF-loaded tokenizers
treat whitespace next to special tokens the way llama.cpp does, not the
way HF does — load the `tokenizer.json` if that distinction matters to
you.
