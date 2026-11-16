#!/usr/bin/env python3
"""Embed bytes as a C++ initializer (no compiler-specific #embed support needed)."""
import sys
from pathlib import Path

source, destination = map(Path, sys.argv[1:])
data = source.read_bytes()
destination.parent.mkdir(parents=True, exist_ok=True)
temporary = destination.with_suffix('.tmp')
with temporary.open('w') as output:
    for offset in range(0, len(data), 32):
        output.write(','.join(map(str, data[offset:offset + 32])) + ',\n')
temporary.replace(destination)
