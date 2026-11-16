#!/usr/bin/env python3
"""Exercise the real installer offline, replacing only the remote download and host detection."""
import hashlib
import io
import os
from pathlib import Path
import subprocess
import tarfile
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / 'install.sh'


class InstallerTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.mock = self.root / 'mock'
        self.mock.mkdir()
        self.home = self.root / 'home'
        self.home.mkdir()
        self.downloads = self.root / 'downloads'
        self.downloads.mkdir()
        self.scratch = self.root / 'scratch'
        self.scratch.mkdir()
        self.env = dict(os.environ, HOME=str(self.home), PATH=str(self.mock) + ':' + os.environ['PATH'],
                        FIXTURE=str(self.downloads), TMPDIR=str(self.scratch))
        for key in ('TACHYON_VERSION', 'TACHYON_INSTALL_DIR', 'TACHYON_CPU'):
            self.env.pop(key, None)
        self.tool('curl', '''#!/bin/bash
set -eu
printf '%s\\n' "$*" >> "$FIXTURE/requests"
[[ ${FAIL_DOWNLOAD:-0} == 0 ]] || exit 22
while (($#)); do
    if [[ $1 == -o ]]; then output=$2; shift 2; else url=$1; shift; fi
done
case $url in
    *.sha256) cp "$FIXTURE/checksum" "$output" ;;
    *.tar.gz) cp "$FIXTURE/archive" "$output" ;;
    *) exit 22 ;;
esac
''')
        self.tool('uname', '#!/bin/bash\ncase $1 in -s) echo "${TEST_OS:-Linux}";; -m) echo "${TEST_ARCH:-x86_64}";; esac\n')
        self.tool('getconf', '#!/bin/bash\n[[ ${TEST_LIBC:-2.38} != musl ]] || exit 1\necho "glibc ${TEST_LIBC:-2.38}"\n')
        self.release()

    def tool(self, name, body):
        file = self.mock / name
        file.write_text(body)
        file.chmod(0o755)

    def release(self, version='1.0.0', floor='2.28', broken=False):
        files = {'VERSION': version + '\n', 'GLIBC_MIN': floor + '\n',
                 'bin/tachyon': '#!/bin/bash\n' + ('exit 1\n' if broken else 'echo ' + version + '\n'),
                 'bin/tachyon-client': '#!/bin/bash\necho client\n'}
        files.update({'libexec/tachyon-' + cpu: '#!/bin/bash\nexit 0\n' for cpu in ('scalar', 'avx2', 'avx512')})
        with tarfile.open(self.downloads / 'archive', 'w:gz') as archive:
            for name, body in files.items():
                data = body.encode()
                info = tarfile.TarInfo('tachyon-release/' + name)
                info.mode = 0o755 if name.startswith(('bin/', 'libexec/')) else 0o644
                info.size = len(data)
                archive.addfile(info, io.BytesIO(data))
        digest = hashlib.sha256((self.downloads / 'archive').read_bytes()).hexdigest()
        (self.downloads / 'checksum').write_text(digest + '  tachyon-linux-x86_64.tar.gz\n')

    def run_installer(self, *arguments, valid=True, pipe=False):
        command = ['bash', str(SCRIPT), *arguments]
        if pipe:
            command = ['bash', '-c', 'cat "$1" | bash', '_', str(SCRIPT)]
        result = subprocess.run(command, env=self.env, capture_output=True, text=True)
        self.assertEqual(result.returncode == 0, valid, result.stdout + result.stderr)
        self.assertEqual(list(self.scratch.iterdir()), [], 'temporary downloads leaked')
        return result.stdout + result.stderr

    def test_piped_install_and_upgrade(self):
        self.run_installer(pipe=True)
        command = self.home / '.local/bin/tachyon'
        self.assertEqual(subprocess.check_output([str(command)], text=True).strip(), '1.0.0')
        old = command.resolve()
        self.release('1.0.1')
        self.run_installer('--version', '1.0.1')
        self.assertNotEqual(command.resolve(), old)
        self.assertTrue(old.exists(), 'running installations must survive upgrades')
        self.assertEqual(subprocess.check_output([str(command)], text=True).strip(), '1.0.1')
        self.assertIn('/releases/download/v1.0.1/', (self.downloads / 'requests').read_text())

    def test_custom_prefix_with_spaces_and_pinned_version(self):
        prefix = self.root / 'custom prefix'
        self.env.update(TACHYON_INSTALL_DIR=str(prefix), TACHYON_VERSION='v1.0.0')
        self.run_installer()
        self.assertTrue((prefix / 'bin/tachyon-client').is_file())
        self.assertIn('/releases/download/v1.0.0/', (self.downloads / 'requests').read_text())

    def test_checksum_failure_preserves_installation(self):
        self.run_installer()
        current = self.home / '.local/lib/tachyon/current'
        target = current.resolve()
        (self.downloads / 'checksum').write_text('0' * 64 + '  archive\n')
        self.assertIn('SHA-256 verification failed', self.run_installer(valid=False))
        self.assertEqual(current.resolve(), target)

    def test_network_failure(self):
        self.env['FAIL_DOWNLOAD'] = '1'
        self.assertIn('release not available', self.run_installer(valid=False))
        self.assertFalse((self.home / '.local/bin/tachyon').exists())

    def test_existing_command_is_preserved(self):
        command = self.home / '.local/bin/tachyon'
        command.parent.mkdir(parents=True)
        command.write_text('mine')
        self.assertIn('not managed', self.run_installer(valid=False))
        self.assertEqual(command.read_text(), 'mine')

    def test_old_glibc_and_musl(self):
        self.env['TEST_LIBC'] = '2.27'
        self.assertIn('requires glibc >= 2.28', self.run_installer(valid=False))
        self.env['TEST_LIBC'] = 'musl'
        self.assertIn('musl/Alpine', self.run_installer(valid=False))

    def test_unsupported_platform(self):
        self.env['TEST_ARCH'] = 'aarch64'
        self.assertIn('x86-64', self.run_installer(valid=False))
        self.env['TEST_OS'] = 'Darwin'
        self.assertIn('supports Linux', self.run_installer(valid=False))
        self.assertFalse((self.downloads / 'requests').exists())

    def test_invalid_arguments_and_help_do_not_download(self):
        for args in [('--version',), ('--prefix', ''), ('--version', '../../bad'), ('--unknown',)]:
            self.run_installer(*args, valid=False)
        self.assertIn('no sudo', self.run_installer('--help'))
        self.assertFalse((self.downloads / 'requests').exists())

    def test_bad_binary_or_metadata_preserves_installation(self):
        self.release(broken=True)
        self.assertIn('cannot run', self.run_installer(valid=False))
        self.release()
        self.assertIn('does not match', self.run_installer('--version', '1.0.1', valid=False))
        self.assertFalse((self.home / '.local/bin/tachyon').exists())


if __name__ == '__main__':
    unittest.main()
