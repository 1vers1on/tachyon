#!/usr/bin/env python
"""Layer-by-layer comparison of Tachyon CPU residuals against HF transformers.
  TACH_DUMP_X=/tmp/tx TACH_PPL_CHUNK=64 ./tachyon ppl -m M.tach -f data/wiki.test.raw -n 1 --prompt-len 64 --device cpu
  .venv/bin/python tools/hf_compare.py <hf_dir> /tmp/tx [--n 64] [--bos]
Prints per layer the relative L2 error of each row (worst rows listed with their tokens)."""
import argparse, numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
ap = argparse.ArgumentParser(); ap.add_argument("model"); ap.add_argument("prefix"); ap.add_argument("--n", type=int, default=64)
ap.add_argument("--bos", action="store_true", help="window starts with <bos> (as tachyon ppl does for BOS models)")
ap.add_argument("-f", default="data/wiki.test.raw")
a = ap.parse_args()
tok = AutoTokenizer.from_pretrained(a.model); m = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32).eval()
ids = tok(open(a.f, encoding="utf-8").read(), add_special_tokens=False).input_ids[:a.n]
if a.bos: ids = [tok.bos_token_id] + ids[1:]
with torch.no_grad(): o = m(torch.tensor([ids]), output_hidden_states=True)
L = len(o.hidden_states) - 1
for l in range(L - 1):   # the last HF hidden state is after the final norm
    t = np.fromfile(f"{a.prefix}.{l}", dtype=np.float32).reshape(a.n, -1); h = o.hidden_states[l + 1][0].numpy()
    r = np.linalg.norm(t - h, axis=1) / np.linalg.norm(h, axis=1)
    worst = np.argsort(-r)[:3]
    print(f"layer {l:3d}: mean rel {r.mean():.4f} max {r.max():.4f}  worst " + ", ".join(f"{i}:{tok.decode([ids[i]])!r}={r[i]:.3f}" for i in worst))
