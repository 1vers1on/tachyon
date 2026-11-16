# Changelog

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
