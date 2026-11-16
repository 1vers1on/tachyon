#!/usr/bin/env python3
"""Compare checkpoint-probe/reference float32 logits with stable KL metrics.

Example: python tools/compare_logits.py baseline.f32 candidate.f32 --ids prompt.ids --max-kl 0.0003
Requires numpy. Both files must contain the same token rows and vocabulary.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def compare(reference, candidate, rows):
    a = np.fromfile(reference, dtype='<f4')
    b = np.fromfile(candidate, dtype='<f4')
    if rows <= 0 or not a.size or a.size != b.size or a.size % rows:
        raise ValueError('logit files must have equal, nonempty shapes divisible by the token count')
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('nonfinite logits')
    a = a.reshape(rows, -1).astype(np.float64)
    b = b.reshape(rows, -1).astype(np.float64)
    def log_softmax(x):
        x = x - x.max(axis=1, keepdims=True)
        return x - np.log(np.exp(x).sum(axis=1, keepdims=True))
    la, lb = log_softmax(a), log_softmax(b)
    kl = np.maximum(0, (np.exp(la) * (la - lb)).sum(axis=1))
    return dict(rows=rows, vocab=a.shape[1], mean_kl=float(kl.mean()), max_kl=float(kl.max()),
                argmax_agreement=float((a.argmax(axis=1) == b.argmax(axis=1)).mean()),
                max_abs_error=float(np.abs(a-b).max()))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('reference', type=Path)
    p.add_argument('candidate', type=Path)
    count = p.add_mutually_exclusive_group(required=True)
    count.add_argument('--ids', type=Path, help='int32 token input to checkpoint-probe')
    count.add_argument('--rows', type=int)
    p.add_argument('--max-kl', type=float, help='fail if mean KL exceeds this tolerance')
    args = p.parse_args()
    if args.ids and args.ids.stat().st_size % 4:
        p.error('token ID file size must be a multiple of four')
    rows = args.ids.stat().st_size // 4 if args.ids else args.rows
    result = compare(args.reference, args.candidate, rows)
    print(json.dumps(result, indent=2))
    return int(args.max_kl is not None and result['mean_kl'] > args.max_kl)


if __name__ == '__main__':
    raise SystemExit(main())
