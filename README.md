# Tachyon

Version **1.0.0**. See the [release notes](CHANGELOG.md); `./tachyon --version` prints the installed version.

A from-scratch LLM inference engine and server in C++, with hand-written CUDA, AVX-512 and AVX2 kernels (plus a scalar fallback).

Tachyon takes a Hugging Face checkpoint, quantizes it to its own `.tach` format, and serves it over
OpenAI-, Anthropic- and Ollama-compatible text generation APIs, with tool calling, so coding agents such as Claude Code can
run on a local model. It also has an ollama-style runner (`tachyon run qwen3:8b`) that downloads,
converts and loads models by name.

It does not use PyTorch, llama.cpp, cuBLAS or any other inference library. Everything from the
safetensors reader and the BPE tokenizer to the tensor-core GEMM and flash attention is in this
repository.

## Highlights

- **Fast single-GPU inference.** Decode is memory-bandwidth bound, and the kernels get close to the
  card's bandwidth: Qwen3-8B (Q4) reaches about 97% of it on the reference RTX 3060 (see
  [Performance](#performance)). A draft model raises that by roughly 1.6x on prose and 2.6x on code.
- **Continuous batching** over KV slots, with prefix caching, chunked prefill and an 8-bit KV cache.
- **Speculative decoding**, either from n-grams (prompt lookup) or from a small draft model. This also
  works for recurrent and hybrid models (Qwen3.5, LFM2), whose state is rolled back when drafts are
  rejected.
- **Importance-weighted 4-bit quantization** (imatrix from calibration text), plus Q8 and an experimental
  Q3 format. Perplexity is checked against Hugging Face transformers for each supported family.
- **MoE models larger than VRAM**: experts that don't fit stay in system RAM and run on the CPU, so
  models such as Qwen3-30B-A3B run on cards with far less memory than their weights.
- **Agent-ready API**: OpenAI `/v1/chat/completions`, Anthropic `/v1/messages` and Ollama `/api/chat` / `/api/generate`, with streaming,
  tool calls in each model's native format, reasoning/thinking blocks and `tool_choice`. A request is
  cancelled when its client disconnects.
- **No build-time CUDA compiler.** The kernels are compiled at runtime with NVRTC, with every model
  dimension baked in as a compile-time constant, and the result is cached.

## Install

On Linux x86-64 (including WSL2):

```bash
curl -fsSL https://raw.githubusercontent.com/1vers1on/tachyon/main/install.sh | bash
```

This downloads a prebuilt release, verifies its SHA-256 checksum, and installs `tachyon` and
`tachyon-client` in `~/.local/bin`. No sudo, compiler, CUDA toolkit or AVX-512 CPU is needed.
If that directory is outside your `PATH`, the installer prints the line to add to your shell profile.

```bash
# Pin a version, or choose an installation prefix (containing bin/ and lib/).
curl -fsSL https://raw.githubusercontent.com/1vers1on/tachyon/main/install.sh | bash -s -- --version 1.0.0
curl -fsSL https://raw.githubusercontent.com/1vers1on/tachyon/main/install.sh | bash -s -- --prefix "$HOME/apps/tachyon"
```

Run the same command to upgrade. Both commands switch to the new release together; previous releases
stay in `~/.local/lib/tachyon` so processes already running can finish. To uninstall, remove the two
symlinks in `~/.local/bin` and `~/.local/lib/tachyon`. Models in `~/.tachyon` are stored separately.

## Requirements

- **Linux x86-64 with glibc 2.28 or newer** for official binaries: Debian 10+, Ubuntu 20.04+,
  Rocky/Alma/RHEL 8+, and newer Fedora/openSUSE systems. Alpine/musl, ARM and macOS do not have
  prebuilt releases. The installer checks the actual glibc requirement recorded in each archive.
- **Any x86-64 CPU.** Releases include scalar, AVX2 + FMA + F16C, and AVX-512 F/BW/VL/VNNI binaries.
  A baseline launcher checks CPU and OS support and selects the fastest supported variant on each run.
  `TACHYON_CPU=scalar|avx2|avx512` selects a specific variant; unsupported selections fail with a message.
- **GPU inference (optional):** an NVIDIA Ampere or newer GPU (compute capability 8.0+) and a compatible
  NVIDIA driver. Releases bundle NVRTC 12.6 and its matching headers; the driver comes from your system.
  `--device auto` uses the CPU when CUDA is unavailable. For GPUs needing a newer toolkit, set
  `CUDA_PATH=/path/to/cuda` (or `TACHYON_CUDA`) to use its compiler and headers instead.
  `TACHYON_NVRTC=/path/to/libnvrtc.so` overrides just the compiler library.
- **Model downloads:** system OpenSSL (`libssl.so.3` or `1.1`) and CA certificates. Certificates are verified
  against the system CA bundle (`SSL_CERT_FILE` overrides it). The installer itself uses Bash, curl, tar,
  gzip and `sha256sum` (or `shasum`).

## Build

Source builds need GCC 11+ with C++20 support, Make, Python 3, and CUDA toolkit headers
(`cuda.h`, `nvrtc.h`). `nvcc` and an NVIDIA GPU are not needed to compile.

```bash
make -j16                    # portable baseline x86-64; builds ./tachyon and ./tachyon-client
make -j16 CPU=avx2           # explicit AVX2/FMA/F16C target
make -j16 CPU=avx512         # explicit AVX-512 VNNI target
make -j16 CPU=native         # tune only for this machine; do not distribute this build
```

`CUDA=/path/to/cuda` selects the build-time toolkit. CPU variants use separate object directories;
changes to compiler flags also trigger rebuilding. Only the C/C++ runtime is linked, and CUDA libraries
are loaded at runtime. For a source build to use the GPU, provide system NVRTC and matching CUDA
headers. The first start on a new model compiles its kernels and caches them in `~/.cache/tachyon`.

See [release builds](docs/releases.md) to make an installer-compatible archive locally or in GitHub Actions.

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

`tachyon run NAME` starts a background model manager when none is running, and logs that startup to
stderr. Its output goes to `$TACHYON_HOME/logs/server.log`. Require an explicitly started server with
`tachyon run NAME --no-autostart`, or set `TACHYON_AUTOSTART=0` in your shell. The opt-out still lets
the CLI connect to an existing manager.

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

### Ollama clients

Point an Ollama client at `http://127.0.0.1:28657` (or start Tachyon with `--port 11434`).
Both `tachyon serve` and `tachyon serve -m FILE --alias NAME` support `/api/chat`, `/api/generate`,
`/api/show`, `/api/tags`, `/api/ps` and `/api/version`. The model manager also supports pulling and
deleting local models. Model names refer to Tachyon's catalog or imported `.tach` files.

```bash
curl localhost:28657/api/chat -d '{
  "model": "qwen3:8b",
  "messages": [{"role": "user", "content": "hi"}],
  "stream": false,
  "think": false,
  "options": {"temperature": 0.7, "num_predict": 128}
}'
```

Streaming defaults to true and returns newline-delimited JSON, following the
[Ollama chat](https://docs.ollama.com/api/chat) and
[generation](https://docs.ollama.com/api/generate) response shapes. Responses include token counts and
durations in nanoseconds. Chat supports `tools`, assistant `tool_calls` with object arguments,
tool results using `tool_name`, and separate `thinking` output. Generate supports `system` and `raw`.
`format: "json"` and schema objects add prompt instructions; output is not grammar constrained.

Supported `options`: `temperature`, `top_k`, `top_p`, `min_p`, `typical_p`, `seed`, `repeat_penalty`,
`repeat_last_n`, `presence_penalty`, `frequency_penalty`, `penalize_prompt`, `logit_bias`, `stop`,
`ignore_eos`, `num_predict`, `num_ctx`, `cache_prompt`, and `draft_num_predict`.
`num_predict: -1` / `-2` generate up to the context limit; zero evaluates the prompt without generating.
`num_ctx` may limit a request within the server's configured context; oversized prompts return HTTP 400.

With the model manager, `keep_alive` accepts seconds or a duration such as `"5m"`; a negative value
keeps a model loaded. Zero unloads it after the response completes. An empty prompt or empty message
list preloads the model; combine it with `keep_alive: 0` to unload. A server started with `-m` keeps its
model loaded for its process lifetime.

This covers text inference, rather than every Ollama feature. Embeddings, images, logprobs, custom
templates, fill-in-the-middle suffixes, legacy `context` tokens, and string-valued thinking levels are
not supported. Other Ollama runtime options do not change the backend's server configuration.

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
| Qwen3 (dense and MoE) | `qwen3:0.6b` … `qwen3:14b`, `qwen3:32b`, `qwen3:30b-a3b` |
| Qwen3.5 (Gated DeltaNet hybrid) | `qwen3.5:0.8b`, `2b`, `4b`, `9b` |
| Qwen2 / Qwen2.5 | `qwen2.5:0.5b` … `32b`, `qwen2.5-coder:7b` … `32b`, `deepseek-r1:1.5b`, `7b` |
| Gemma 3 / Gemma 4 | `gemma3:270m`, `gemma4:e2b`, `e4b`, `12b` |
| Gemma (first generation) | `gemma:2b`, `gemma:7b` (gated) |
| LFM2 / LFM2.5 (short-conv hybrid, MoE) | `lfm2.5:350m` … `2.6b`, `lfm2.5:8b-a1b` |
| OLMoE | `olmoe:1b-7b` |
| Llama / Mistral | `mistral:7b` (Llama 3.x is wired in but untested) |
| Mixtral | `mixtral:8x7b`, `mixtral:8x22b` (large MoE models; require substantial system RAM) |
| SmolLM3 | `smollm3:3b` |
| Granite 3.x | `granite3.3:2b`, `8b` |
| Phi-3 / Phi-4-mini | `phi4-mini` |

Run `./tachyon catalog` for the full list. Dense models need their quantized weights plus KV cache to fit
in VRAM, so the 14B models want roughly 10 GB and the 32B models roughly 20-24 GB (MoE models can spill
experts to system RAM). Each family's perplexity was checked against Hugging Face
transformers, or CPU against CUDA for MoE models that are too large for a reference run.

Gemma (first generation) and Mixtral are validated with tiny random checkpoints against Transformers
logits, including both packed and original per-expert Mixtral weights. Full-size quality and throughput
have not been measured for these additions. Mistral/Mixtral sliding-attention windows are preserved.

## Performance

Reference numbers from one machine (RTX 3060 12 GB, Ryzen 7 7700X, WSL2), with greedy decoding and
Q4 + imatrix models unless noted. Decode speed scales roughly with memory bandwidth, so expect different
numbers on other hardware.

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

### RTX 5090

Measured with `tools/bench.sh gpu` on an RTX 5090: a 2048-token prompt, 128 generated tokens, greedy decoding with
speculation off, median of 3 runs (min-max). Decode is single-stream.

| Model | Prefill tok/s | Decode tok/s |
|---|---|---|
| Qwen3-0.6B Q4 | 73920 (73888-74238) | 923 (923-924) |
| Qwen3-8B Q4 | 9256 (9255-9258) | 228 (228-228) |

### CPU kernel sets

The CPU backend has AVX-512, AVX2 and scalar kernels (see [Requirements](#requirements)). Median of 5 interleaved
runs on the Ryzen 7 7700X (Zen 4) above, with `--device cpu -t 8`, a 512-token prompt, 128 generated tokens and
greedy decoding. Each set is built from the same source with a different `-march`: native, `haswell` and `x86-64`.

| Model | Kernels | Prefill tok/s | Decode tok/s |
|---|---|---|---|
| Qwen3-0.6B Q4 | AVX-512 | 425 | 115 |
| | AVX2 | 432 | 117 |
| | scalar | 51 | 23 |
| Qwen3-0.6B Q8 | AVX-512 | 383 | 67 |
| | AVX2 | 380 | 68 |
| | scalar | 83 | 32 |
| Qwen3-1.7B Q4 | AVX-512 | 170 | 46 |
| | AVX2 | 175 | 47 |
| | scalar | 23 | 11 |

AVX-512 and AVX2 are within 3% of each other here, which is inside the run-to-run spread (about 10%), so this
machine doesn't separate them. Zen 4 executes AVX-512 on 256-bit datapaths, so a CPU with full-width 512-bit
units may behave differently; that has not been measured. The scalar kernels are 4 to 8x slower on Q4 and 2 to 5x
slower on Q8, so use them only on CPUs that lack AVX2.

### Benchmark your own hardware

`tools/bench.sh` pulls the models it needs, runs `tachyon bench` several times (interleaved, so drift in machine
state hits every configuration equally) and prints the median and range as a markdown table. It needs a built
tree (see [Build](#build)) and `python3`.

```bash
tools/bench.sh gpu                                  # GPU backend: qwen3:0.6b and qwen3:8b, 2048-token prompt, 3 runs
tools/bench.sh gpu qwen3:8b qwen3.5:9b mistral:7b   # any catalog names or Hugging Face repos
tools/bench.sh cpu                                  # CPU kernel sets: qwen3:0.6b, qwen3:0.6b-q8, qwen3:1.7b, 512-token prompt, 5 runs
tools/bench.sh cpu -r 3 -t 16 -p 2048 qwen3:1.7b    # runs, CPU threads, prompt length, generated tokens (-n)
```

Speculative decoding is off in these runs (`-s N` turns it on). `tachyon bench` uses a prompt that repeats one
sentence, so with n-gram speculation (on by default on the GPU) nearly every draft token is accepted and decode
looks 4 to 5 times faster than real single-stream decode. As a sanity check, single-stream decode cannot exceed
memory bandwidth divided by the model file size (roughly 1.8 TB/s / 4.8 GB, about 370 tok/s, for Qwen3-8B Q4 on an
RTX 5090).

Models are downloaded from Hugging Face and converted (Q4 + importance matrix) the first time they are used, and
are kept in `$TACHYON_HOME` (default `~/.tachyon`). Gated models need `HF_TOKEN`. `cpu` mode builds a binary per
kernel set in `build/bench/` (AVX-512, AVX2 and scalar, whichever the CPU can run), so it takes a few minutes the
first time. The raw runs are written to `build/bench/`.

For a single run on one model, use `./tachyon bench -m model.tach [-n 256] [--prompt-len 512] [--device cpu|cuda]`.
`tachyon show <name>` prints the `.tach` file of a pulled model.

### CUDA sampling kernels

The vocabulary maximum used by candidate sampling is reduced across 32 GPU blocks per row, and
candidate writes reserve space per warp. Overflow falls back to full logits so high-temperature
sampling keeps the correct distribution. On an RTX 3060, the two sampling kernels together took
11.7 µs instead of 19.4 µs for one 151,936-token vocabulary row, and 16.3 µs instead of 22.4 µs for
four rows (medians of five interleaved runs, 296 candidates per row). These are isolated kernel
measurements; full inference is dominated by the model forward pass and needs separate measurement.

```bash
make build/bench-sampling
build/bench-sampling                       # synthetic sparse and overflowing candidate distributions
build/bench-sampling /tmp/old-kernels.cu    # compare against an earlier kernels.cu revision
```

The benchmark captures 100 pairs of kernels per CUDA graph and times 20 replays with CUDA events,
so host launch overhead does not dominate the comparison.

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

### Inference settings

Sampling flags work with `serve`, `run -m FILE`, `run NAME`, and `bench`. With `serve`, they set the
defaults used when clients omit a field. Start from [sampling.example.json](sampling.example.json):

```bash
# A manager with explicit backend settings and defaults for every model it loads.
./tachyon serve --device cuda -c 8192 --slots 2 --prefill-chunk 512 \
  --sampling-config sampling.example.json --temp 0.6

# Run locally with a fixed seed, a limited repetition window and a stop string.
./tachyon run -m models/qwen3-0.6b-q4.tach -p "Write a short story" \
  --sampling-config sampling.example.json --repeat-last-n 64 --seed 42 --stop "THE END"
```

Config files and flags apply in command-line order; later values replace earlier ones. Repeated
`--stop` flags append stop strings. A request overrides server defaults, which override the model's
generation config. Unknown config keys and invalid values are rejected. Omitted settings preserve
existing behavior; the example file is an optional starting point. Restart or reload a model after
changing a sampling config file. Backend settings such as context, KV precision and batch size remain
startup options, and the manager accepts them directly as shown above.

| Control | CLI | API / config | Behavior |
|---|---|---|---|
| Typical sampling | `--typical-p P` | `typical_p` | Keep probability mass closest to expected information content; `(0, 1]`, 1 disables it |
| Repetition window | `--repeat-last-n N` | `repeat_last_n` | Penalize the last N tokens; -1 uses all history, 0 disables repetition/presence/frequency penalties |
| Repetition strength | `--repeat-penalty P` | `repetition_penalty` / Ollama `repeat_penalty` | Positive multiplier; 1 disables it |
| Presence / frequency | `--presence-penalty P`, `--frequency-penalty P` | `presence_penalty`, `frequency_penalty` | Penalize seen tokens or their counts within the repetition window |
| Prompt penalties | `--no-penalize-prompt` | `penalize_prompt: false` | Apply repetition penalties to generated tokens only |
| Token bias | `--logit-bias '{"123": -10}'` | `logit_bias` | Add -100 through 100 to individual token logits; keys are vocabulary token IDs |
| Maximum output | `--max-tokens N` | `max_tokens` / Ollama `num_predict` | 0 evaluates without generation; -1/-2 use remaining context |
| Stop strings | `--stop TEXT` | `stop` / Anthropic `stop_sequences` | Stop before returning a matched string; an empty array clears inherited stops |
| EOS handling | `--ignore-eos` | `ignore_eos` | Continue through EOS tokens until a token/context limit or stop string |
| Prompt cache | `--no-prompt-cache` | `cache_prompt: false` | Start this request without reusing a cached prompt prefix |
| Speculation | `--speculative-tokens N` | `speculative_tokens` / Ollama `draft_num_predict` | -1 uses server defaults; 0 disables all drafting; 1–8 caps draft depth within server capabilities |
| Seed | `--seed N` | `seed` | -1 selects a random seed; 0 and positive seeds are reproducible |

Temperature zero is greedy decoding and skips probability filters. Otherwise, the filter order is
temperature, top-k, [locally typical sampling](https://arxiv.org/abs/2202.00666), min-p, then top-p.
Token biases and repetition penalties are applied before sampling; bias can require transferring full
logits to the CPU. These controls are extensions on the OpenAI and Anthropic routes; Ollama passes them
inside `options`. OpenAI and Anthropic requests put them at the top level.

`GET /api/settings` (also `/v1/settings`) reports effective generation defaults and backend settings
for a single-model server. On the model manager it reports manager settings, explicit generation
overrides and model-process arguments. It uses the same API-key authentication as inference routes.

### Logging

Log lines go to stderr with a timestamp, a level (`ERROR`, `WARN`, `INFO`, `DEBUG`) and, while a request is
being handled, its id (`req=N`), so the lines of one request can be followed through the server and the engine.
Every request gets one access line (method, path, status, duration, bytes, client address; 4xx is a warning, 5xx
an error) and generation requests add a summary with token counts, speed and time to first token. Headers,
prompts and API keys are never logged.

| Option | Environment | Effect |
|---|---|---|
| `-v` / `-q` | | Debug level / warnings and errors only |
| `--log-level L` | `TACHYON_LOG_LEVEL` | `error`, `warn`, `info` (default) or `debug` |
| `--log-file PATH` | `TACHYON_LOG_FILE` | Also append every line to `PATH` (several processes can share one file) |
| `--log-format text\|json` | `TACHYON_LOG_FORMAT` | `json` writes one JSON object per line (`ts`, `level`, `pid`, `req`, `msg`) |

Flags override the environment. Colors are used on a terminal unless `NO_COLOR` is set. The model server
(`tachyon serve` without `-m`) passes `-v` / `-q` on to the model processes it starts, whose output is in
`$TACHYON_HOME/logs/<model>.log`.

## How it works

```
HF checkpoint ──convert──▶ .tach (Q4/Q8/Q3 weights, tokenizer, chat template, layer graph)
                                │ mmap
                                ▼
HTTP (OpenAI / Anthropic / Ollama) ─▶ server ─▶ engine (continuous batching, prefix cache, speculation)
                                              │
                                              ▼
                                  backend: CUDA (NVRTC kernels) or CPU (AVX-512 / AVX2 / scalar)
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
tools/          benchmark script, regression and API test scripts, Hugging Face reference comparisons
data/           calibration text (wikitext-2 train) and evaluation text (wikitext-2 test)
```

## Testing

```bash
make test                                # offline file validation, CPU kernels and HTTP/API regressions
make test-cuda                           # CUDA candidate sampling vs full logits, including overflow
TACH_TEST_DEVICE=cuda python3 tests/settings_test.py ./tachyon  # inference controls on CUDA
make test-models PYTHON=.venv/bin/python  # tiny model conversion + CPU logits vs Transformers; no downloads
TACH_TEST_DEVICE=cuda make test-models PYTHON=.venv/bin/python  # quantized references on CUDA
tools/regress.sh                          # perplexity regression across models, both CUDA paths
python tools/api_test.py 28657 qwen3:1.7b  # OpenAI + Anthropic SDK tests (pip install openai anthropic)
python tools/tool_test.py 28657 MODEL...   # tool-call round trip and prefix reuse per model
```

`tools/hf_ppl.py` and `tools/hf_compare.py` produce Hugging Face transformers reference outputs. They
need `torch` and `transformers`.
`make test-models` additionally needs `numpy` and `safetensors` in the selected Python environment.
`make test` also checks malformed tokenizer and importance-matrix files, HTTP client framing and stream
completion, CLI failures, authentication, model aliases, and model-process socket isolation. These checks
use temporary fixtures and do not download models. Custom `BUILD` and `OUT` paths are supported by the tests.

## Limitations

- Linux only. The CUDA backend needs an Ampere-or-newer NVIDIA GPU, and the CPU backend is x86-64 only (AVX-512 is fastest, AVX2 is next, scalar works on any x86-64 CPU).
- There is no grammar-constrained decoding, so a model can still emit a malformed tool call. It is then
  returned as text.
- Image inputs, logprobs and `n > 1` aren't supported.
- Chat templates are written per model family, not run from Jinja.

## License

MIT. See [LICENSE.txt](LICENSE.txt).
