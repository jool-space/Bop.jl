```@meta
CurrentModule = Bop
```

# Bop

Pure-Julia tokenization for language models, loading HuggingFace
`tokenizer.json` files or GGUF metadata directly. See the
[README](https://github.com/jool-space/Bop.jl) for scope and
correctness methodology.

Names are `public`, not exported: either qualify (`Bop.encode`) or
import explicitly (`using Bop: Tokenizer, encode, decode`).

## Loading

```@docs
Tokenizer
from_file
from_pretrained
from_json
```

## Encoding and decoding

```@docs
encode
decode
Encoding
encode_batch
decode_batch
```

## GGUF

```@docs
from_gguf
gguf_metadata
```

GGUF names its pre-tokenizer instead of embedding the split patterns,
so loading consults a verified name table.

```@docs
PRE_TOKENIZERS
PreSpec
```
