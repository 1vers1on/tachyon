#!/usr/bin/env python
"""HF transformers reference perplexity, matching `tachyon ppl` (independent windows, positions win/4..win-2
scored, <bos> at position 0 of every window when the tokenizer adds one).
usage: .venv/bin/python tools/hf_ppl.py <hf_dir> [-n windows] [--win 512] [--dtype float32|bfloat16] [--no-bos | --bos]"""
import os, sys, math, argparse, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
ap = argparse.ArgumentParser()
ap.add_argument("model"); ap.add_argument("-n", type=int, default=2); ap.add_argument("--win", type=int, default=512)
ap.add_argument("--dtype", default="float32"); ap.add_argument("-f", default="data/wiki.test.raw"); ap.add_argument("--no-bos", action="store_true"); ap.add_argument("--bos", action="store_true", help="force <bos>")
a = ap.parse_args()
tok = AutoTokenizer.from_pretrained(a.model)
model = AutoModelForCausalLM.from_pretrained(a.model, dtype=getattr(torch, a.dtype)).eval()
ids = tok(open(a.f, encoding="utf-8").read(), add_special_tokens=False).input_ids
probe = tok("x").input_ids
bos = tok.bos_token_id if (not a.no_bos and tok.bos_token_id is not None and (a.bos or (probe and probe[0] == tok.bos_token_id))) else None
nll, cnt = 0.0, 0
for w in range(a.n):
    x = ids[w * a.win:(w + 1) * a.win]
    if len(x) < a.win: break
    if bos is not None: x = [bos] + x[1:]
    with torch.no_grad():
        lg = model(torch.tensor([x])).logits[0].float()
    lp = torch.log_softmax(lg, -1)
    tgt = ids[w * a.win:(w + 1) * a.win]
    for p in range(a.win // 4, a.win - 1):
        nll -= lp[p, tgt[p + 1]].item(); cnt += 1
        if os.environ.get('DUMP'): print(f'[nll] w{w} p{p} tok{tgt[p+1]} {-lp[p, tgt[p+1]].item():.4f}')
print(f"hf perplexity: {math.exp(nll / cnt):.4f} over {cnt} tokens (bos={bos}, dtype={a.dtype})")
