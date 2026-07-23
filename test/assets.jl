# Fetch the tokenizer.json corpus the fixtures were generated from.
# Assets are gitignored (tens of MB); this downloads any that are missing.
using Downloads

const ASSET_REPOS = Dict(
    "gpt2" => "openai-community/gpt2",
    "qwen2.5-0.5b-instruct" => "Qwen/Qwen2.5-0.5B-Instruct",
    "qwen3-0.6b" => "Qwen/Qwen3-0.6B",
    "qwen3.5" => "Qwen/Qwen3.5-4B",
    "deepseek-v3" => "deepseek-ai/DeepSeek-V3",
    "llama-3.2-1b" => "unsloth/Llama-3.2-1B",
    "gpt-oss-20b" => "openai/gpt-oss-20b",
    "phi-4" => "microsoft/phi-4",
    "modernbert-base" => "answerdotai/ModernBERT-base",
    "mistral-nemo-instruct-2407" => "unsloth/Mistral-Nemo-Instruct-2407",
    "glm-4.5-air" => "zai-org/GLM-4.5-Air",
    "olmo-2-1124-7b" => "allenai/OLMo-2-1124-7B",
)

function ensure_assets(dir::AbstractString)
    for (name, repo) in ASSET_REPOS
        path = joinpath(dir, name, "tokenizer.json")
        isfile(path) && continue
        url = "https://huggingface.co/$repo/resolve/main/tokenizer.json"
        @info "downloading test asset" name url
        mkpath(dirname(path))
        Downloads.download(url, path)
    end
    # GGUF metadata asset: just the header + KV section of the 8 GB
    # Qwen3.5 GGUF (KVs end ~10.9 MB in; Bop's reader never reads past
    # them, so a truncated file is a valid test asset).
    gguf = joinpath(dir, "qwen3.5", "metadata.gguf")
    if !isfile(gguf)
        url = "https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/resolve/main/Qwen3.5-4B-BF16.gguf"
        @info "downloading test asset (12MB range)" gguf
        mkpath(dirname(gguf))
        Downloads.download(url, gguf; headers = ["Range" => "bytes=0-11999999"])
    end
end
