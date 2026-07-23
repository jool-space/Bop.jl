module Bop

using Downloads
using JSON
using StringViews
using Unicode

using Republic: @public

include("bytelevel.jl")

include("model.jl")

include("tokenizer.jl")
@public Tokenizer, Encoding, encode, decode, encode_batch, decode_batch

include("load.jl")
@public from_file, from_json, from_pretrained

include("gguf.jl")
@public from_gguf, gguf_metadata, PRE_TOKENIZERS, PreSpec

end
