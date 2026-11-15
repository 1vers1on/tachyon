"""Offline conversion + CPU logits checks against tiny random Transformers models.

Run with `make test-models PYTHON=.venv/bin/python`. No Hub access or pretrained weights.
Both original per-expert and newer packed Mixtral checkpoint layouts are exercised.
"""
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

import numpy as np
import torch
from safetensors.torch import save_file
from transformers import (GemmaConfig, GemmaForCausalLM, MixtralConfig,
                          MixtralForCausalLM, MistralConfig, MistralForCausalLM)

binary, probe = (str(Path(p).resolve()) for p in sys.argv[1:3])
torch.set_num_threads(2)
torch.manual_seed(1234)


def run(*args):
    p = subprocess.run(args, capture_output=True, text=True)
    assert p.returncode == 0, (args, p.returncode, p.stderr)


def byte_vocab():
    # GPT-2 reversible byte alphabet, without merges.
    visible = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    vocab = {chr(b): b for b in visible}
    for i, b in enumerate(b for b in range(256) if b not in visible):
        vocab[chr(256 + i)] = b
    return vocab


def legacy_mixtral(state):
    result = {}
    for name, value in state.items():
        name = name.replace(".mlp.", ".block_sparse_moe.")
        if name.endswith("experts.gate_up_proj"):
            for e, matrix in enumerate(value):
                gate, up = matrix.chunk(2, dim=0)
                prefix = name.removesuffix("gate_up_proj") + f"{e}."
                result[prefix + "w1.weight"] = gate.contiguous()
                result[prefix + "w3.weight"] = up.contiguous()
        elif name.endswith("experts.down_proj"):
            for e, matrix in enumerate(value):
                result[name.removesuffix("down_proj") + f"{e}.w2.weight"] = matrix.contiguous()
        else:
            result[name] = value.clone()
    return result


def header(path):
    with path.open("rb") as f:
        magic, offset, size = struct.unpack("<8sQQ", f.read(24))
        assert magic == b"TACHYON2"
        f.seek(offset)
        return json.loads(f.read(size))


cases = [
    ("gemma", GemmaConfig, GemmaForCausalLM, {}, False),
    ("mixtral-packed", MixtralConfig, MixtralForCausalLM, {}, False),
    ("mixtral-original", MixtralConfig, MixtralForCausalLM, {}, True),
    ("mixtral-window", MixtralConfig, MixtralForCausalLM, {"sliding_window": 5}, True),
    ("mistral-window", MistralConfig, MistralForCausalLM, {"sliding_window": 5}, False),
]
with tempfile.TemporaryDirectory(prefix="tachyon-models-") as tmp:
    root = Path(tmp)
    for name, config_class, model_class, extra, legacy in cases:
        folder = root / name
        folder.mkdir()
        cfg = config_class(hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                           num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                           vocab_size=256, max_position_embeddings=128,
                           num_local_experts=4, num_experts_per_tok=2, **extra)
        cfg._attn_implementation = "eager"
        model = model_class(cfg).eval()
        # Nonzero Gemma norm weights verify the converter's (1+w) adjustment.
        if name == "gemma":
            with torch.no_grad():
                for key, value in model.named_parameters():
                    if "norm.weight" in key:
                        value.uniform_(-0.2, 0.2)
        state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if legacy:
            state = legacy_mixtral(state)
        save_file(state, folder / "model.safetensors")
        config = cfg.to_dict()
        if legacy:
            config["rope_theta"] = config.pop("rope_parameters")["rope_theta"]
        (folder / "config.json").write_text(json.dumps(config))
        (folder / "tokenizer.json").write_text(json.dumps(dict(
            model=dict(type="BPE", vocab=byte_vocab(), merges=[]), added_tokens=[])))
        (folder / "tokenizer_config.json").write_text("{}")
        ids = torch.tensor([[65 + i % 23 for i in range(20)]])
        with torch.no_grad():
            reference = model(ids).logits[0].numpy()
        for quant in ("f16", "q8"):
            out = folder / f"{quant}.tach"
            run(binary, "convert", str(folder), str(out), "-q", quant, "-t", "2")
            converted = header(out)["config"]
            layers = converted["layers"]
            assert all(layer.get("window", 0) == extra.get("sliding_window", 0) for layer in layers)
            if name.startswith("mixtral"):
                assert all(layer["n_experts"] == 4 and layer["top_k"] == 2 and layer["norm_topk"] for layer in layers)
            elif name == "gemma":
                assert converted["act"] == "gelu" and converted["embed_scale"] == 8
                assert all(not layer.get("sandwich", False) and layer.get("qk_norm", 0) == 0 for layer in layers)
            outputs = []
            for chunk in (1, 4):
                logits = folder / "logits.f32"
                run(probe, str(out), str(logits), str(chunk), "20")
                actual = np.fromfile(logits, dtype=np.float32).reshape(reference.shape)
                rel = np.linalg.norm(actual - reference) / np.linalg.norm(reference)
                limit = 0.002 if quant == "f16" else 0.04
                assert np.isfinite(actual).all() and rel < limit, (name, quant, chunk, rel)
                assert np.max(np.abs(actual - reference)) < (0.002 if quant == "f16" else 0.06)
                outputs.append(actual.copy())
                print(f"{name:18} {quant} chunk={chunk}: relative logits error {rel:.6f}")
            np.testing.assert_allclose(outputs[0], outputs[1], atol=2e-6, rtol=2e-5)
print("model conversion and inference: all checks passed")
