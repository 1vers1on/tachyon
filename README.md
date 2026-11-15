# Tachyon

A from-scratch LLM inference engine and server in C++, with hand-written CUDA and AVX-512 kernels.

Tachyon takes a Hugging Face checkpoint, quantizes it to its own `.tach` format, and serves it over
OpenAI- and Anthropic-compatible HTTP APIs, with tool calling, so coding agents such as Claude Code can
run on a local model. It also has an ollama-style runner (`tachyon run qwen3:8b`) that downloads,
converts and loads models by name.

It does not use PyTorch, llama.cpp, cuBLAS or any other inference library. Everything from the
safetensors reader and the BPE tokenizer to the tensor-core GEMM and flash attention is in this
repository.

## Highlights

- **Fast single-GPU inference.** Qwen3-8B (Q4) decodes at about 68 tok/s on an RTX 3060, which is about
  97% of the card's measured memory bandwidth. With a draft model, that rises to about 110 tok/s on prose
  and 180 tok/s on code.
- **Continuous batching** over KV slots, with prefix caching, chunked prefill and an 8-bit KV cache. Eight
  concurrent streams of the 8B model reach about 430 tok/s in total.
- **Speculative decoding**, either from n-grams (prompt lookup) or from a small draft model. This also
  works for recurrent and hybrid models (Qwen3.5, LFM2), whose state is rolled back when drafts are
  rejected.
- **Importance-weighted 4-bit quantization** (imatrix from calibration text), plus Q8 and an experimental
  Q3 format. Perplexity is checked against Hugging Face transformers for each supported family.
- **MoE models larger than VRAM**: experts that don't fit stay in system RAM and run on the CPU. For
  example, Qwen3-30B-A3B runs at about 50 tok/s on a 12 GB card.
- **Agent-ready API**: OpenAI `/v1/chat/completions` and Anthropic `/v1/messages`, both with streaming,
  tool calls in each model's native format, reasoning/thinking blocks and `tool_choice`. A request is
  cancelled when its client disconnects.
- **No build-time CUDA compiler.** The kernels are compiled at runtime with NVRTC, with every model
  dimension baked in as a compile-time constant, and the result is cached.

## Requirements

- Linux (developed on WSL2).
- `g++` 15 or newer (the CUDA kernels and calibration text are embedded with `#embed`).
- CUDA toolkit headers, `libcuda` and `libnvrtc` (developed with CUDA 12.6). `nvcc` is not used.
- An NVIDIA GPU with compute capability 8.0 or higher (Ampere or newer) for the CUDA backend.
- A CPU with AVX-512 (F/BW/VL/VNNI), such as Zen 4 or Ice Lake and newer. The CPU backend is always
  compiled, so the build currently needs this even if you only use the GPU.
- `curl` for `tachyon pull`.

## Build

```bash
make -j16          # builds ./tachyon and ./tachyon-client
```

Set `CUDA=/path/to/cuda` if the toolkit isn't in `/usr/local/cuda`. The first start on a new model
compiles its kernels, which takes about 2 seconds. They are cached in `~/.cache/tachyon`, so later
starts are instant.

## Quick start

### Run models by name

```bash
./tachyon run qwen3:8b                     # download, convert (Q4 + imatrix), start the server, open a chat
./tachyon run qwen3.5:9b "explain RoPE"    # one-shot prompt
./tachyon serve                            # model server on 127.0.0.1:28657; loads models on demand
./tachyon catalog                          # models available by name
./tachyon list | ps | show M | stop M | rm M
```

Models can be named from the catalog (add `-q8` for 8-bit) or as any Hugging Face repo (`org/name`) of
a supported architecture. Models are stored in `$TACHYON_HOME` (default `~/.tachyon`). An existing
Hugging Face cache (`$HF_HOME/hub`) is reused when it has the model. Gated repos need `HF_TOKEN`.

### Use it from an agent

OpenAI-compatible clients use `http://127.0.0.1:28657/v1`. Anthropic-compatible clients use
`http://127.0.0.1:28657`. For example, Claude Code:

```bash
ANTHROPIC_BASE_URL=http://127.0.0.1:28657 ANTHROPIC_AUTH_TOKEN=local \
ANTHROPIC_MODEL=qwen3.5:9b ANTHROPIC_SMALL_FAST_MODEL=qwen3.5:9b claude
```

Or with curl:

```bash
curl localhost:28657/v1/chat/completions \
  -d '{"model":"qwen3:8b","messages":[{"role":"user","content":"hi"}]}'
```

### Convert and serve one model manually

```bash
# Convert a Hugging Face checkpoint folder (config.json, tokenizer.json, *.safetensors).
# --calib enables importance-weighted 4-bit quantization (recommended).
./tachyon convert ~/hf/Qwen3-1.7B models/qwen3-1.7b-q4.tach --calib data/calib.txt

./tachyon serve -m models/qwen3-1.7b-q4.tach --port 28657

./tachyon-client                     # interactive streaming chat
./tachyon-client -p "Explain RoPE"   # one-shot
./tachyon-client --bench 16 -c 4     # load test
```

## Supported models

| Family | Examples in the catalog |
|---|---|
| Qwen3 (dense and MoE) | `qwen3:0.6b` … `qwen3:14b`, `qwen3:30b-a3b` |
| Qwen3.5 (Gated DeltaNet hybrid) | `qwen3.5:0.8b`, `2b`, `4b`, `9b` |
| Qwen2 / Qwen2.5 | `qwen2.5:0.5b` … `7b`, `qwen2.5-coder:7b`, `deepseek-r1:1.5b`, `7b` |
| Gemma 3 / Gemma 4 | `gemma3:270m`, `gemma4:e2b`, `e4b`, `12b` |
| Gemma (first generation) | `gemma:2b`, `gemma:7b` (gated) |
| LFM2 / LFM2.5 (short-conv hybrid, MoE) | `lfm2.5:350m` … `2.6b`, `lfm2.5:8b-a1b` |
| OLMoE | `olmoe:1b-7b` |
| Llama / Mistral | `mistral:7b` (Llama 3.x is wired in but untested) |
| Mixtral | `mixtral:8x7b`, `mixtral:8x22b` (large MoE models; require substantial system RAM) |
| SmolLM3 | `smollm3:3b` |
| Granite 3.x | `granite3.3:2b`, `8b` |
| Phi-3 / Phi-4-mini | `phi4-mini` |

Run `./tachyon catalog` for the full list. Each family's perplexity was checked against Hugging Face
transformers, or CPU against CUDA for MoE models that are too large for a reference run.

Gemma (first generation) and Mixtral are validated with tiny random checkpoints against Transformers
logits, including both packed and original per-expert Mixtral weights. Full-size quality and throughput
have not been measured for these additions. Mistral/Mixtral sliding-attention windows are preserved.

## Performance

Measured on an RTX 3060 12 GB with a Ryzen 7 7700X under WSL2. Models are Q4 + imatrix unless noted,
with greedy decoding.

| Model | Decode tok/s (single stream) | Prefill tok/s (2048 tokens) |
|---|---|---|
| Qwen3-0.6B | 528 | — |
| Qwen3-8B | 68 (≈110–180 with `--draft qwen3-0.6b`) | 1.5k |
| Qwen3.5-9B | 63 (≈140 on code with a draft model) | 1.5k |
| Gemma-4-12B-it | 41 | 0.98k |
| Mistral-7B-Instruct-v0.3 | 68 | 1.25k |
| Qwen3-30B-A3B (half of the experts in RAM) | 50 | 1.66k |

On the CPU backend alone (8 threads), Qwen3-1.7B Q4 decodes at about 52 tok/s.
With vectorized CPU attention, Qwen3-0.6B Q4 measures about 324 tok/s prefill and 75 tok/s decode
after a 2048-token prompt (8 threads, 64 generated tokens, speculation off; two runs).

To measure on your own hardware, run `./tachyon bench -m model.tach`.

## Command reference

| Command | Purpose |
|---|---|
| `tachyon convert DIR OUT.tach [-q q4\|q8] [--calib FILE]` | Convert and quantize a Hugging Face checkpoint |
| `tachyon serve -m M.tach` | Serve one model |
| `tachyon serve` | Model manager: serves any local model by name |
| `tachyon run -m M.tach -p "text"` | Generate locally and print speed stats |
| `tachyon bench -m M.tach` | Prefill and decode benchmark |
| `tachyon ppl -m M.tach -f data/wiki.test.raw` | Perplexity, to check quantization quality |
| `tachyon tokenize -m M.tach -p "text"` | Show token ids |
| `tachyon selftest` | Check the CPU kernels against a reference |

Useful options:

| Option | Effect |
|---|---|
| `--device auto\|cuda\|cpu` | Select the backend |
| `-c N` | Context length per slot |
| `--slots N` | Number of concurrent sequences |
| `--spec N` | Number of n-gram speculative tokens (0 turns it off) |
| `--draft small.tach` | Draft model for speculative decoding (must share the tokenizer) |
| `--kv8` | 8-bit KV cache: half the KV memory, small quality cost |
| `--cpu-moe N` | Keep the experts of N MoE layers in system RAM (default: automatic) |
| `--api-key KEY` | Require an API key (Bearer or `x-api-key`) |

Run `./tachyon --help` for everything.

## How it works

```
HF checkpoint ──convert──▶ .tach (Q4/Q8/Q3 weights, tokenizer, chat template, layer graph)
                                │ mmap
                                ▼
HTTP (OpenAI / Anthropic) ─▶ server ─▶ engine (continuous batching, prefix cache, speculation)
                                              │
                                              ▼
                                  backend: CUDA (NVRTC kernels) or CPU (AVX-512)
```

- **Decode** uses int8 activations and `dp4a` GEMVs, with RMSNorm, RoPE, SiLU and quantization fused into
  neighboring kernels. Each decode step is one CUDA graph. Small batches (2–8 tokens, used for speculation
  and batching) go through a tensor-core GEMV that costs about 1.2× a single token.
- **Prefill** uses fp16 tensor-core GEMMs with Q4 dequantization in shared memory, FlashAttention-2-style
  attention, and chunked kernels for the delta rule and convolutions of hybrid models.
- **Model graph**: each layer is described by a `LayerCfg` (attention, short convolution or DeltaNet
  mixer; dense or MoE FFN). The converter normalizes architectural quirks so the runtime stays generic.

## Repository layout

```
src/            engine, backends, server, converter, tokenizer, runner, client
src/kernels/    all CUDA kernels (embedded into the binary)
tools/          regression and API test scripts, Hugging Face reference comparisons
data/           calibration text (wikitext-2 train) and evaluation text (wikitext-2 test)
```

## Testing

```bash
make test                                # offline file validation + CPU kernel reference checks
make test-models PYTHON=.venv/bin/python  # tiny model conversion + CPU logits vs Transformers; no downloads
TACH_TEST_DEVICE=cuda make test-models PYTHON=.venv/bin/python  # quantized references on CUDA
tools/regress.sh                          # perplexity regression across models, both CUDA paths
python tools/api_test.py 28657 qwen3:1.7b  # OpenAI + Anthropic SDK tests (pip install openai anthropic)
python tools/tool_test.py 28657 MODEL...   # tool-call round trip and prefix reuse per model
```

`tools/hf_ppl.py` and `tools/hf_compare.py` produce Hugging Face transformers reference outputs. They
need `torch` and `transformers`.
`make test-models` additionally needs `numpy` and `safetensors` in the selected Python environment.

## Limitations

- Linux only. The CUDA backend needs an Ampere-or-newer NVIDIA GPU, and the build needs an AVX-512 CPU.
- There is no grammar-constrained decoding, so a model can still emit a malformed tool call. It is then
  returned as text.
- Image inputs, logprobs and `n > 1` aren't supported.
- Chat templates are written per model family, not run from Jinja.
- Ollama's management endpoints (`/api/tags`, `/api/ps`, `/api/pull`, …) exist, but `/api/chat` and
  `/api/generate` don't.

## License

MIT. See [LICENSE.txt](LICENSE.txt).
