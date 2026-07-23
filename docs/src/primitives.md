```@meta
CurrentModule = Bop
```

# Primitives

The building blocks behind [`from_gguf`](@ref): GGUF names its
pre-tokenizer instead of embedding the split patterns, so loading
consults a verified name table.

```@docs
PRE_TOKENIZERS
PreSpec
```
