# Bop

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://jool-space.github.io/Bop.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://jool-space.github.io/Bop.jl/dev/)
[![Build Status](https://github.com/jool-space/Bop.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/jool-space/Bop.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/jool-space/Bop.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/jool-space/Bop.jl)

Pure-Julia tokenizer for language models. Loads HuggingFace `tokenizer.json`
files directly — no Python, no Rust, no dependencies beyond JSON3.

```julia
using Bop

tok = Tokenizer("tokenizer.json")
enc = encode(tok, "Hello, world!")
enc.ids           # Vector{Int} (0-based ids, as HF)
enc.tokens        # materialized lazily
decode(tok, enc.ids)
encode(tok, text; add_special_tokens = false)
decode(tok, ids; skip_special_tokens = false)
```

## Scope

Byte-level BPE — the tokenizer family used by essentially every current
open-weights LM (GPT-2/4 lineage, Qwen, Llama 3/4, DeepSeek, and friends).
Supported `tokenizer.json` components:

- model: `BPE` (incl. `ignore_merges`, both merges formats)
- normalizers: `NFC`/`NFD`/`NFKC`/`NFKD`, `Sequence`
- pre-tokenizers: `Split` (Isolated/Removed, `invert`, chained), `ByteLevel`
  (incl. `add_prefix_space` and the built-in GPT-2 `use_regex` pattern)
- post-processors: `TemplateProcessing` (single-sequence), `ByteLevel`
- added/special tokens incl. `lstrip`/`rstrip`, `ByteLevel` decoder

Patterns are rewritten at load to match Oniguruma's character classes
(HF's regex engine) where PCRE2 disagrees — e.g. U+180E, whitespace in
PCRE2 but not in Oniguruma since Unicode 6.3.

Anything outside this errors loudly at load — nothing mis-tokenizes
silently. Sentencepiece-converted files (`byte_fallback`, e.g. Gemma,
Llama 2) are not yet supported.

## Correctness

Differentially tested against HF `tokenizers` (the Rust library) over
twelve tokenizers — GPT-2, Qwen 2.5 / 3 / 3.5, DeepSeek-V3, Llama 3.2,
GPT-OSS, Phi-4, Mistral-Nemo, GLM-4.5, OLMo-2, ModernBERT — on a battery
of adversarial cases (unicode whitespace zoo, astral letters, contraction
casing, embedded special tokens, format chars, number classes, …) plus
randomized fuzzing: ids, token strings (incl. lstrip/rstrip surface
forms), and both decode modes match exactly. Fixtures are
regenerated with `uv run --with tokenizers python3 scripts/gen_fixtures.py`.

Throughput (Qwen3.5, 6 MB mixed text): ~12.5 MB/s single-threaded (~4×
HF sequential), ~55 MB/s on 8 threads (~4× HF's fully parallel
`encode_batch`; one `Tokenizer` per thread — the BPE cache is not
thread-safe). Prompt-scale encode is ~3.5 µs. `encode` also accepts raw
`AbstractVector{UInt8}` buffers (e.g. mmap) copy-free via StringViews.

GGUF: `Bop.from_gguf(path_or_metadata)` loads `gpt2`-model (byte-level
BPE) tokenizers straight from GGUF metadata — the pre-tokenizer name →
pattern table (`Bop.PRE_TOKENIZERS`) covers qwen35/qwen2/llama-bpe/gpt-2,
each pinned by tests to its family's tokenizer.json.
