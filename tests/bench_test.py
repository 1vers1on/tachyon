"""Run the shipped benchmark through an install symlink, with no source tree or build tools."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

SCRIPT = Path(sys.argv[1] if len(sys.argv) > 1 else 'tools/bench.sh').resolve()
sys.argv[1:] = []


class BenchTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='tachyon bench ')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        release = self.root / 'release'
        (release / 'bin').mkdir(parents=True)
        (release / 'libexec').mkdir()
        shutil.copy2(SCRIPT, release / 'bin/tachyon-bench')
        self.command = self.root / 'tachyon-bench'
        self.command.symlink_to(release / 'bin/tachyon-bench')
        model = self.root / 'model with spaces.tach'
        model.touch()
        fake = '''#!/bin/bash
set -eu
case $1 in
pull) echo "pull $2" >> "$CALLS" ;;
show) printf '  file           %s (1 MB)\\n' "$MODEL" ;;
bench)
    [[ $2 == -m && $3 == "$MODEL" ]] || exit 1
    n=$(cat "$COUNTER" 2>/dev/null || echo 0); n=$((n+1)); echo "$n" > "$COUNTER"
    echo "prompt = $((n*10)) tok/s | gen = $((n*2)) tok/s"
    ;;
*) exit 1 ;;
esac
'''
        for binary in [release / 'bin/tachyon', *(release / 'libexec' / ('tachyon-' + x)
                                                 for x in ('scalar', 'avx2', 'avx512'))]:
            binary.write_text(fake)
            binary.chmod(0o755)
        mock = self.root / 'mock'
        mock.mkdir()
        for tool in ('make', 'g++', 'python3'):
            p = mock / tool
            p.write_text('#!/bin/bash\necho "unexpected build dependency" >&2\nexit 99\n')
            p.chmod(0o755)
        self.env = dict(os.environ, PATH=str(mock) + ':' + os.environ['PATH'],
                        MODEL=str(model), CALLS=str(self.root / 'calls'), COUNTER=str(self.root / 'counter'),
                        TACHYON_BENCH_DIR=str(self.root / 'results'))

    def run_bench(self, *args, valid=True):
        p = subprocess.run([str(self.command), *args], cwd='/', env=self.env,
                           capture_output=True, text=True, timeout=30)
        self.assertEqual(p.returncode == 0, valid, p.stdout + p.stderr)
        return p

    def test_gpu_even_and_odd_medians(self):
        for reps, expected in [('3', '20 (10-30) | 4 (2-6)'), ('4', '25 (10-40) | 5 (2-8)')]:
            Path(self.env['COUNTER']).unlink(missing_ok=True)
            p = self.run_bench('gpu', '-r', reps, 'qwen3:0.6b')
            self.assertIn('| qwen3:0.6b | cuda | ' + expected + ' |', p.stdout)
        self.assertTrue(list((self.root / 'results').glob('gpu-*.log')))

    def test_cpu_uses_bundled_variants(self):
        p = self.run_bench('cpu', '-r', '1', 'qwen3:0.6b')
        self.assertIn('| scalar |', p.stdout)

    def test_help_and_invalid_options(self):
        self.assertIn('benchmark', self.run_bench('--help').stdout)
        for args in [('gpu', '-r', '0'), ('gpu', '-p', '-1'), ('gpu', '-s', '9'), ('bad',)]:
            self.run_bench(*args, valid=False)
        self.assertFalse(Path(self.env['CALLS']).exists())


if __name__ == '__main__':
    unittest.main()
