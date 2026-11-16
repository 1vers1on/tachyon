# Changelog

## Unreleased

RTX 3060 12 GB model set. New weight formats are recorded by exact name in the `.tach` file and are never
requantized on import:

- Add `PQ2_0`, `PTQ1_0` (Prism ternary, with block Hadamard transforms and signs), `Q4_0`, `Q8_0`, `Q4_K`, `Q5_K`, `Q6_K` and `BF16`
  tensor types, a GGUF importer (`qwen35` and `gemma4`), and a Tachyon-defined `q4_k_m` recipe (`tachyon-q4_k_m-v1`).
- Add catalog entries `bonsai2:9b`, `bonsai2:9b-ptq1_0`, `bonsai2:27b`, `mimo2.6:9b`, `gemma4:12b-qat`, `k2-horizon:7b`
  and `mellum2.1:12b-a2.5b`; GGUF entries download one named file and import it.
- Add the K2 Horizon dense architecture (grouped RMSNorm) and the Mellum MoE architecture (per-layer sliding window
  and per-layer YaRN frequencies); checkpoint Jinja chat templates are rendered natively (vendored from llama.cpp, MIT).
- CUDA: packed-format models (ternary, Q4_0/Q8_0, K-quants, F16/BF16/F32 matrices) now use the normal decode path:
  CUDA graphs, multi-slot decode batches, MoE decode and speculative verification, with new block-wise GEMV kernels
  that keep fp32 activations. Decode on an RTX 3060 went from 4-24 tok/s to 14-88 tok/s without speculation.
- Fix `INCS` being expanded before `BUILD` was set, which broke a plain `make test`.
- CPU backend: apply the per-layer YaRN attention factor (it was only applied on CUDA) and copy 16-bit KV
  entries correctly when duplicating a slot.

## 1.0.0 — 2026-11-16

First stable release of the C++ inference engine, model converter, named-model runner and
OpenAI-, Anthropic- and Ollama-compatible text APIs. See the README for supported models,
build requirements and known limitations.

Release fixes:

- Add a checksum-verified installer and relocatable Linux release archives with automatic CPU selection,
  bundled NVRTC/headers, and a glibc 2.28 CI baseline across multiple distros.
- Default source builds to baseline x86-64, offer explicit AVX2/AVX-512 targets, and replace GCC 15-only
  resource embedding with generated initializers so GCC 11+ can build the project.
- Reject invalid device names and invalid benchmark sizes; return a failure exit code when local generation fails.
- Stop at the earliest matching stop string when a decoded token contains multiple matches.
- Validate tokenizer blobs, token IDs, model graph dimensions, weight shapes and importance-matrix bounds before use.
- Reject malformed JSON numbers and unescaped control characters.
- Detect malformed or truncated HTTP responses and incomplete generation streams in the CLI clients.
- Share response framing between the CLI and downloader, handle chunk extensions and trailers, and enforce socket timeouts.
- Restrict download credentials to the original origin and report the release version in the download user agent.
- Resolve catalog names ending in `:latest`, and authenticate named-model CLI commands using `TACHYON_API_KEY`.
- Prevent model children from inheriting manager sockets, and perform manager shutdown outside signal handlers.
- Require the correct HTTP methods for model pulling and deletion.
- Avoid reading unfinished request results during disconnect logging; make greedy candidate ties deterministic.
- Add `tachyon --version` and extend the offline regression suite, including custom build/output paths.

The `.tach` on-disk format version is unchanged. The stricter validation rejects inconsistent model metadata.

Validation completed on Linux x86-64 with GCC 15 and an RTX 3060: the offline regression suite,
CPU and CUDA tiny-model comparisons against Transformers, CUDA sampling and inference controls,
AVX2 and scalar kernel/quantizer checks, and AddressSanitizer/UndefinedBehaviorSanitizer checks
for file, text and HTTP parsing. Perplexity smoke checks completed for the local Qwen3, Qwen3.5,
LFM2.5 and Gemma 3 checkpoints. Full-size testing of every catalog model remains outside this coverage.
