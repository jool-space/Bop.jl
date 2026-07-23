module Bop

using Downloads
using JSON
using StringViews
using Unicode

public Tokenizer, Encoding, encode, decode, encode_batch, decode_batch,
    from_file, from_json, from_pretrained, from_gguf, gguf_metadata,
    PRE_TOKENIZERS, PreSpec

include("bytelevel.jl")
include("model.jl")
include("tokenizer.jl")
include("load.jl")
include("gguf.jl")

end
