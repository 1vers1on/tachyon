#!/usr/bin/env python3
"""Check dispatch with masked CPUs, then run the actual kernels under the same masks."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

launcher = Path(sys.argv[1]).resolve()
env = dict(os.environ)
env.pop('TACHYON_CPU', None)
with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    (root / 'bin').mkdir()
    (root / 'libexec').mkdir()
    shutil.copy2(launcher, root / 'bin/tachyon')
    for cpu in ('scalar', 'avx2', 'avx512'):
        file = root / 'libexec' / ('tachyon-' + cpu)
        file.write_text('#!/bin/sh\necho ' + cpu + '\n')
        file.chmod(0o755)
    for mask, expected in [('Nehalem', 'scalar'), ('Haswell', 'avx2')]:
        command = ['qemu-x86_64', '-cpu', mask, str(root / 'bin/tachyon')]
        result = subprocess.run(command, env=env, capture_output=True, text=True, check=True)
        assert result.stdout.strip() == expected, (mask, result)
        bad = subprocess.run(command, env=dict(env, TACHYON_CPU='avx512'), capture_output=True, text=True)
        assert bad.returncode and 'does not support' in bad.stderr, bad
        binary = launcher.parent.parent / 'libexec' / ('tachyon-' + expected)
        subprocess.run(['qemu-x86_64', '-cpu', mask, str(binary), 'selftest'], env=env, check=True)
    bad = subprocess.run([str(root / 'bin/tachyon')], env=dict(env, TACHYON_CPU='invalid'), capture_output=True, text=True)
    assert bad.returncode and 'must be' in bad.stderr, bad
print('Launcher dispatch, unsupported ISA rejection, and scalar/AVX2 kernels passed')
