# Environment Variables Reference

Complete reference of all environment variables supported by this vLLM serverless worker.

## Model Configuration

| Variable | Default | Description |
|---|---|---|
| `MODEL_NAME` | *(required)* | Hugging Face model name or path. Alias for `MODEL`. |
| `MODEL_REVISION` | — | Model revision (branch/tag/commit) to load. Alias for `REVISION`. |
| `TOKENIZER_NAME` | same as model | Tokenizer repo to use a different tokenizer than the model's default. Alias for `TOKENIZER`. |
| `TOKENIZER_REVISION` | — | Tokenizer revision to load. |
| `TOKENIZER_MODE` | `auto` | Tokenizer mode. Options: `auto`, `slow`. |
| `SKIP_TOKENIZER_INIT` | `false` | Skip initialization of tokenizer and detokenizer. |
| `TRUST_REMOTE_CODE` | `false` | Trust remote code from Hugging Face. |
| `DOWNLOAD_DIR` | — | Directory to download and load the weights. |
| `BASE_PATH` | `/runpod-volume` | Storage directory for Hugging Face cache and model. |
| `HF_TOKEN` | — | Hugging Face access token (used during Docker build for gated models). |
| `CUSTOM_CHAT_TEMPLATE` | — | Custom Jinja chat template override. |

## Quantization

| Variable | Default | Description |
|---|---|---|
| `QUANTIZATION` | — | Quantization method. Options: `awq`, `squeezellm`, `gptq`, `None`. |
| `LOAD_FORMAT` | `auto` | Model weights format. Options: `auto`, `pt`, `safetensors`, `npcache`, `dummy`, `tensorizer`, `bitsandbytes`. When set to `bitsandbytes`, also sets `quantization=bitsandbytes`. |

## Engine Configuration

| Variable | Default | Description |
|---|---|---|
| `DTYPE` | `auto` | Data type for model weights and activations. Options: `auto`, `half`, `float16`, `bfloat16`, `float`, `float32`. |
| `MAX_MODEL_LEN` | auto-detected | Model context length. Set to `0` to use model's default. |
| `MAX_NUM_BATCHED_TOKENS` | `max_model_len` | Maximum number of batched tokens per iteration. Set to `0` to auto-resolve. |
| `MAX_NUM_SEQS` | `256` | Maximum number of sequences per iteration. |
| `MAX_LOGPROBS` | `20` | Max number of log probs to return. |
| `GPU_MEMORY_UTILIZATION` | `0.95` | GPU VRAM utilization fraction. |
| `BLOCK_SIZE` | `16` | Token block size for contiguous chunks of tokens. |
| `SWAP_SPACE` | `4` | CPU swap space size (GiB) per GPU. |
| `SEED` | `0` | Random seed for operations. |
| `ENFORCE_EAGER` | `false` | Always use eager-mode PyTorch (disables CUDA graphs). |
| `MAX_SEQ_LEN_TO_CAPTURE` | `8192` | Max sequence length for CUDA graph capture. |
| `CPU_OFFLOAD_GB` | `0` | CPU offload size in GB. |
| `ENABLE_PREFIX_CACHING` | `false` | Enable automatic prefix caching. |
| `DISABLE_SLIDING_WINDOW` | `false` | Disable sliding window, capping to sliding window size. |
| `ENABLE_CHUNKED_PREFILL` | `false` | Enable chunked prefill requests. |
| `GUIDED_DECODING_BACKEND` | `outlines` | Backend for guided/structured decoding. |
| `KV_CACHE_DTYPE` | `auto` | Data type for KV cache storage. Options: `auto`, `fp8`. |
| `DEVICE` | `auto` | Device type for vLLM execution. Options: `auto`, `cuda`, `neuron`, `cpu`, `openvino`, `tpu`, `xpu`. |
| `ATTENTION_BACKEND` | — | Attention backend override (replaces deprecated `VLLM_ATTENTION_BACKEND`). |
| `STREAM_INTERVAL` | `1` | Token stream interval. |

## Parallelism & Distribution

| Variable | Default | Description |
|---|---|---|
| `TENSOR_PARALLEL_SIZE` | `1` (auto-set to GPU count if >1) | Number of tensor parallel replicas. |
| `PIPELINE_PARALLEL_SIZE` | `1` | Number of pipeline stages. |
| `DISTRIBUTED_EXECUTOR_BACKEND` | `mp` | Backend for distributed serving. Options: `ray`, `mp`. |
| `MAX_PARALLEL_LOADING_WORKERS` | — | Load model sequentially in multiple batches. Forced to `None` when >1 GPU detected. |
| `DISABLE_CUSTOM_ALL_REDUCE` | `false` | Disable custom all reduce kernel. |
| `RAY_WORKERS_USE_NSIGHT` | `false` | Use nsight to profile Ray workers. |
| `ENABLE_EXPERT_PARALLEL` | `false` | Enable Expert Parallel for MoE models. |

## LoRA Configuration

| Variable | Default | Description |
|---|---|---|
| `ENABLE_LORA` | `false` | Enable handling of LoRA adapters. |
| `MAX_LORAS` | `1` | Max number of LoRAs in a single batch. |
| `MAX_LORA_RANK` | `16` | Max LoRA rank. |
| `LORA_DTYPE` | `auto` | Data type for LoRA. Options: `auto`, `float16`, `bfloat16`, `float32`. |
| `LORA_EXTRA_VOCAB_SIZE` | `256` | Extra vocabulary size for LoRA. |
| `MAX_CPU_LORAS` | — | Maximum number of LoRAs to store in CPU memory. |
| `FULLY_SHARDED_LORAS` | `false` | Enable fully sharded LoRA layers. |
| `LORA_MODULES` | `[]` | JSON array of LoRA adapter configs. Example: `[{"name": "adapter1", "path": "/path"}]`. |

## Prompt Adapter Configuration

| Variable | Default | Description |
|---|---|---|
| `ENABLE_PROMPT_ADAPTER` | `false` | Enable prompt adapters. |
| `MAX_PROMPT_ADAPTERS` | `1` | Max number of prompt adapters. |
| `MAX_PROMPT_ADAPTER_TOKEN` | `0` | Max prompt adapter token count. |

## Speculative Decoding

| Variable | Default | Description |
|---|---|---|
| `SPECULATIVE_CONFIG` | — | Full speculative decoding config as a JSON string. Overrides all individual speculative env vars. |
| `SPECULATIVE_METHOD` | — | Method to use. Options: `draft_model`, `ngram`, `eagle`, `eagle3`, `medusa`, `mlp_speculator`. |
| `SPECULATIVE_MODEL` | — | Draft model name/path. Auto-detects method from model name if `SPECULATIVE_METHOD` is not set. |
| `NUM_SPECULATIVE_TOKENS` | — | Number of speculative tokens to sample from the draft model. |
| `NGRAM_PROMPT_LOOKUP_MAX` | — | Max window size for n-gram prompt lookup. |
| `NGRAM_PROMPT_LOOKUP_MIN` | — | Min window size for n-gram prompt lookup. |
| `SPECULATIVE_DRAFT_TENSOR_PARALLEL_SIZE` | — | Tensor parallel size for the draft model. |
| `SPECULATIVE_MAX_MODEL_LEN` | — | Max model length for the draft model. |
| `SPECULATIVE_DISABLE_BY_BATCH_SIZE` | — | Disable speculative decoding when batch size exceeds this value. |
| `SPECULATIVE_QUANTIZATION` | — | Quantization method for the draft model. |
| `SPECULATIVE_MODEL_REVISION` | — | Revision for the draft model. |
| `SPECULATIVE_ENFORCE_EAGER` | — | Enforce eager mode for the draft model. |
| `SPEC_DECODING_ACCEPTANCE_METHOD` | `rejection_sampler` | Acceptance method for speculative decoding. |

## RunPod Worker Settings

| Variable | Default | Description |
|---|---|---|
| `MAX_CONCURRENCY` | `30` | Max concurrent requests per worker. vLLM has an internal queue, so this is for scaling/load balancing efficiency. |
| `DEFAULT_BATCH_SIZE` | `50` | Default and maximum batch size for token streaming to reduce HTTP calls. |
| `MIN_BATCH_SIZE` | `1` | Starting batch size for the first request (grows by growth factor). |
| `BATCH_SIZE_GROWTH_FACTOR` | `3` | Growth factor for dynamic batch size. Batch grows as `min * factor^n` up to max. |

## OpenAI Compatibility Settings

| Variable | Default | Description |
|---|---|---|
| `OPENAI_SERVED_MODEL_NAME_OVERRIDE` | model name | Overrides the served model name in OpenAI API responses. |
| `OPENAI_RESPONSE_ROLE` | `assistant` | Role of the LLM response in chat completions. |
| `RAW_OPENAI_OUTPUT` | `true` (`1`) | Return raw OpenAI-formatted output instead of just text. Accepts `true`/`false` or `1`/`0`. |
| `TRUST_REQUEST_CHAT_TEMPLATE` | `false` | Trust chat templates provided in requests. |
| `ENABLE_AUTO_TOOL_CHOICE` | `false` | Enable automatic tool choice in chat completions. |
| `EXCLUDE_TOOLS_WHEN_TOOL_CHOICE_NONE` | `false` | Exclude tools from prompt when tool_choice is none. |
| `TOOL_CALL_PARSER` | — | Tool call parser. Options: `hermes`, `mistral`, `llama3_json`, `pythonic`, `internlm`. |
| `REASONING_PARSER` | — | Parser for reasoning-capable models. Options: `deepseek_r1`, `qwen3`, `granite`, `hunyuan_a13b`. |
| `RETURN_TOKENS_AS_TOKEN_IDS` | `false` | Return tokens as token IDs in the response. |
| `ENABLE_PROMPT_TOKENS_DETAILS` | `false` | Include prompt token details in usage response. |
| `ENABLE_FORCE_INCLUDE_USAGE` | `false` | Always include usage info in streaming responses. |
| `ENABLE_LOG_OUTPUTS` | `false` | Log model outputs. |
| `LOG_ERROR_STACK` | `false` | Log full error stack traces. |

## Logging

| Variable | Default | Description |
|---|---|---|
| `DISABLE_LOG_STATS` | `false` | Disable logging statistics. |
| `ENABLE_LOG_REQUESTS` | `false` | Enable vLLM request logging. |

## Multimodal

| Variable | Default | Description |
|---|---|---|
| `LIMIT_MM_PER_PROMPT` | — | Limit multimodal inputs per prompt. Format: `image=1,video=0`. |

## Tokenizer Pool

| Variable | Default | Description |
|---|---|---|
| `TOKENIZER_POOL_SIZE` | `0` | Tokenizer pool size for async tokenization. |
| `TOKENIZER_POOL_TYPE` | `ray` | Tokenizer pool backend type. |

## Docker Build Args

These are only used at build time, not runtime:

| Variable | Default | Description |
|---|---|---|
| `VLLM_VERSION` | `0.18.0` | vLLM version to install. |
| `VLLM_NIGHTLY` | `false` | Install nightly vLLM build (overwrites pinned version). |
| `MODEL_NAME` | — | Bake model into Docker image during build. |
| `TOKENIZER_NAME` | — | Bake tokenizer into image. |
| `HF_TOKEN` | — | HF token for downloading gated models during build. |
| `BASE_PATH` | `/runpod-volume` | Base storage path in container. |
| `QUANTIZATION` | — | Quantization to apply during model download. |
| `MODEL_REVISION` | — | Model revision to download. |
| `TOKENIZER_REVISION` | — | Tokenizer revision to download. |

## Container Environment (set in Dockerfile)

These are set automatically and generally should not be overridden:

| Variable | Value | Description |
|---|---|---|
| `HF_DATASETS_CACHE` | `${BASE_PATH}/huggingface-cache/datasets` | HF datasets cache directory. |
| `HUGGINGFACE_HUB_CACHE` | `${BASE_PATH}/huggingface-cache/hub` | HF hub cache directory. |
| `HF_HOME` | `${BASE_PATH}/huggingface-cache/hub` | HF home directory. |
| `HF_HUB_ENABLE_HF_TRANSFER` | `0` | Disable hf_transfer for downloads. |
| `RAY_METRICS_EXPORT_ENABLED` | `0` | Suppress Ray metrics agent warnings. |
| `RAY_DISABLE_USAGE_STATS` | `1` | Disable Ray usage stats collection. |
| `TOKENIZERS_PARALLELISM` | `false` | Prevent rayon thread pool panic in containers. |
| `RAYON_NUM_THREADS` | `4` | Limit rayon threads to avoid ulimit issues. |
| `PYTHONPATH` | `/:/vllm-workspace` | Python module search path. |

## Auto-Discovery

Any field in vLLM's `AsyncEngineArgs` dataclass can be set as an environment variable using its **UPPERCASED** name. For example:
- `max_model_len` field -> `MAX_MODEL_LEN` env var
- `rope_scaling` field -> `ROPE_SCALING` env var (JSON string)

This means new vLLM engine parameters are automatically supported without code changes.

## Deprecated Variables

| Deprecated Variable | Replacement | Notes |
|---|---|---|
| `MAX_CONTEXT_LEN_TO_CAPTURE` | `MAX_SEQ_LEN_TO_CAPTURE` | Logs a deprecation warning. |
| `VLLM_ATTENTION_BACKEND` | `ATTENTION_BACKEND` | Maps to `--attention-backend` CLI arg. |
| `DISABLE_LOG_REQUESTS` | `ENABLE_LOG_REQUESTS` | Inverted boolean logic. |
| `kv_cache_dtype=fp8_e5m2` | `kv_cache_dtype=fp8` | Auto-converted with warning. |

## Type Conventions

- **Booleans**: `true`/`false`, `1`/`0`, `yes`/`no`, `on`/`off`
- **JSON objects/arrays**: Pass as JSON strings (e.g., `LORA_MODULES='[{"name":"a","path":"/p"}]'`)
- **Comma-separated lists**: For simple lists and tuples (e.g., `LIMIT_MM_PER_PROMPT=image=1,video=0`)
- **Numbers**: Plain integer or float strings
