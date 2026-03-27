# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Summary

RunPod serverless worker providing OpenAI-compatible LLM inference endpoints powered by vLLM 0.16.0. Deploys any Hugging Face model as a drop-in OpenAI API replacement on RunPod's serverless infrastructure.

## Build & Run

This is a Docker-based project with no local test suite. All testing happens on the RunPod platform.

```bash
# Build the Docker image (Option 1: runtime model download)
docker buildx bake --file docker-bake.hcl

# Build with a baked-in model (Option 2: model in image)
docker buildx bake --file docker-bake.hcl --set "*.args.MODEL_NAME=org/model-name" --set "*.args.HF_TOKEN=hf_xxx"

# Run locally (requires GPU)
docker run --gpus all -e MODEL_NAME=org/model-name -e HF_TOKEN=hf_xxx <image>
```

Build variables are defined in `docker-bake.hcl`: `DOCKERHUB_REPO`, `DOCKERHUB_IMG`, `RELEASE_VERSION`.

Test definitions live in `.runpod/tests_json` and run on the RunPod platform against a small model (`HuggingFaceTB/SmolLM2-135M-Instruct`).

## Architecture

### Request Flow

```
RunPod Job → handler.py → JobInput parsing → Engine selection → vLLM generation → Streaming response
```

`handler.py` is the entry point (`python3 /src/handler.py`). It routes requests to one of two engines based on whether the request has an `openai_route`:

- **`vLLMEngine`** (`engine.py`): Base engine wrapping vLLM's `AsyncLLMEngine`. Handles tokenization, dynamic batching, and streaming generation.
- **`OpenAIvLLMEngine`** (`engine.py`): Extends `vLLMEngine` with OpenAI API compatibility — serves `/v1/chat/completions`, `/v1/completions`, and `/v1/models`. Defers initialization to first request for event loop compatibility. Handles LoRA adapter loading, tool call parsing, and reasoning parsers.

### Configuration System (`engine_args.py`)

The configuration system auto-discovers vLLM engine args from environment variables:

1. Scans all `AsyncEngineArgs` fields and checks for UPPERCASED env var equivalents
2. Applies backward-compat aliases (`MODEL_NAME` → `model`, `TOKENIZER_NAME` → `tokenizer`)
3. Merges with `/local_model_args.json` (written during Docker build for baked models)
4. Handles special cases: speculative decoding config, `limit_mm_per_prompt` parsing, multi-GPU tensor parallelism detection, HF overrides rope_scaling sanitization

Custom defaults differ from vLLM defaults (e.g., `gpu_memory_utilization=0.95`, `max_num_seqs=256`). These are defined in the `CustomDefaults` class.

**Critical:** When changing defaults, always update `.runpod/hub.json` to keep the Hub UI in sync.

### Key Modules

- `utils.py`: `JobInput` (request parsing), `BatchSize` (dynamic batch growth from min→max via growth factor), `DummyRequest`/`DummyState` (mock objects for OpenAI engine)
- `tokenizer.py`: `TokenizerWrapper` — detects/applies chat templates, supports `CUSTOM_CHAT_TEMPLATE` env var override. Mistral models use vLLM's native tokenizer instead.
- `constants.py`: `DEFAULT_BATCH_SIZE=50`, `DEFAULT_MAX_CONCURRENCY=30`, `DEFAULT_BATCH_SIZE_GROWTH_FACTOR=3`, `DEFAULT_MIN_BATCH_SIZE=1`
- `download_model.py`: Runs during Docker build to download and cache model weights, creates `/local_model_args.json`

### Environment Variable Conventions

- vLLM settings: uppercase versions of vLLM parameter names (auto-discovered)
- RunPod settings: `MAX_CONCURRENCY`, `DEFAULT_BATCH_SIZE`, etc.
- OpenAI settings: `OPENAI_` prefix (`OPENAI_SERVED_MODEL_NAME_OVERRIDE`, `OPENAI_RESPONSE_ROLE`)
- Feature flags: `ENABLE_*` / `DISABLE_*` pattern
- Complex types: JSON strings for dicts/lists; comma-separated for simple lists
- Booleans: `'true'`/`'false'` or `0`/`1`

## CI/CD

- **Release builds:** Triggered by git tags matching `v*.*.*` → pushes versioned Docker image
- **Dev builds:** Triggered by PRs → pushes `dev-<branch-name>` image
- Release process: tag from main branch (`git tag 2.8.0 && git push origin 2.8.0`)

## Important Design Decisions

- The handler guards against re-initialization in vLLM worker subprocesses (`if __name__ == "__main__"` plus `RUNPOD_POD_ID` check)
- CUDA errors trigger `sys.exit(1)` for pod restart rather than graceful recovery
- OpenAI engine initialization is deferred to first request to ensure event loop availability
- Dynamic batch sizing uses exponential growth: `min_batch_size * growth_factor^n` up to `max_batch_size`
- Input tokens are counted once and cached to avoid redundant tokenization
