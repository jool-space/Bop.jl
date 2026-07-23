# Bop

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://docs.jool.space/Bop.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://docs.jool.space/Bop.jl/dev/)
[![Build Status](https://github.com/jool-space/Bop.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/jool-space/Bop.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/jool-space/Bop.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/jool-space/Bop.jl)

Tokenize text for language models in pure Julia. Bop reads the same
`tokenizer.json` files that HuggingFace models ship, or the metadata
inside a GGUF file.

```julia
using Bop

tok = Tokenizer("tokenizer.json")       # or Bop.from_pretrained("Qwen/Qwen3-0.6B")
enc = encode(tok, "Hello, world!")
enc.ids                                 # 0-based ids, exactly as HF
enc.tokens                              # token strings, computed on demand
decode(tok, enc.ids)                    # "Hello, world!"
```

This covers the tokenizers used by essentially every current open-weights
LM — byte-level BPE and sentencepiece-converted BPE. Unigram (T5) and
WordPiece (BERT) models are unsupported and error at load, as does any
component outside the supported set: files never mis-tokenize silently.

- `Bop.from_gguf("model.gguf")` builds the tokenizer from GGUF metadata,
  with no `tokenizer.json` sidecar.
- A `Tokenizer` is immutable and safe to share across tasks; encoding is
  ~3.5 µs per chat prompt, ~12.5 MB/s per thread on bulk text.
- `encode` also accepts raw `AbstractVector{UInt8}` buffers, copy-free.
- `Bop.encode_batch` / `Bop.decode_batch`, `add_special_tokens`,
  `skip_special_tokens` behave as in HF.

Correctness is defined as matching the HF `tokenizers` library exactly
and enforced differentially: seventeen real tokenizers, adversarial
fixtures, and randomized fuzzing (`test/scripts/`) — ids, token strings,
and both decode modes. One known divergence: GGUF cannot carry
added-token `lstrip`/`rstrip` flags, so GGUF-loaded tokenizers treat
whitespace next to special tokens as llama.cpp does; load the
`tokenizer.json` if that matters. Offsets and word ids are out of scope.
