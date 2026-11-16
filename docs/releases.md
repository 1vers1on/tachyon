# Building and publishing releases

The installer expects two assets on each GitHub release:

```
tachyon-linux-x86_64.tar.gz
tachyon-linux-x86_64.tar.gz.sha256
```

The archive includes a baseline CPU launcher, three inference binaries (scalar, AVX2, AVX-512 VNNI),
a baseline `tachyon-client`, NVRTC and its builtins, matching CUDA headers, license notices, and
version/commit/glibc metadata. All binaries use explicit CPU flags. A GPU and AVX-512 hardware are
unnecessary on the build machine: CUDA kernels are compiled on the user's machine by NVRTC.
The host's NVIDIA driver is never bundled.

## GitHub Actions

`.github/workflows/release.yml` builds on pushes to `main`, pull requests, version tags and manual
dispatches. It uses CUDA 12.6.3 on Rocky Linux 8 with GCC Toolset 12, links the C++ runtimes statically,
and checks that the resulting glibc minimum is 2.28. The offline tests run before packaging.
The archive is tested on Debian, Ubuntu, Rocky, Fedora and openSUSE without a toolkit or NVIDIA driver.
QEMU tests CPU selection and the scalar/AVX2 kernels with CPU features masked.

Every successful build uploads the two assets as the `linux-x86_64` workflow artifact. For a release:

1. Update `TACH_VERSION` in `src/common.h` and the changelog, then commit.
2. Push a matching tag, for example `git tag v1.0.1 && git push origin v1.0.1`.
3. After the tests pass, the workflow creates the GitHub release and attaches both assets.

The installer uses GitHub's latest release by default, or `--version 1.0.1` for a pinned version.
Checksums detect corrupted downloads and a release changing mid-download; they are integrity checks,
not independent publisher signatures. The initial release must exist before the install command works.

## Build on your own machine

With GCC 11+, Make, Python 3, binutils and a CUDA toolkit installed:

```bash
CUDA=/usr/local/cuda JOBS=8 tools/package.sh
# Optional explicit version; it must match src/common.h:
tools/package.sh 1.0.0
```

The files appear in `dist/`. This uses the same archive format and CPU flags as CI. It records the
actual minimum glibc required by the binaries and bundled CUDA libraries. Building on a newer distro
can raise that minimum, even with static C++ runtimes. Build inside the CI container to retain the
official glibc 2.28 baseline:

```bash
docker run --rm -v "$PWD:/src" -w /src \
  nvidia/cuda:12.6.3-devel-rockylinux8 \
  bash -lc 'dnf install -y git gcc-toolset-12-gcc-c++ gcc-toolset-12-libstdc++-devel make python3.11 tar gzip binutils findutils curl &&
    ln -sf /usr/bin/python3.11 /usr/local/bin/python3 &&
    source /opt/rh/gcc-toolset-12/enable && mkdir -p build &&
    curl -fsSL https://docs.nvidia.com/cuda/archive/12.6.3/eula/index.html -o build/CUDA-EULA.html &&
    CUDA_EULA=build/CUDA-EULA.html tools/package.sh'
```

Use a clean checkout when building published assets; `COMMIT` records the checkout's HEAD.
Keep `TACH_VERSION` and the GitHub tag aligned. To upload locally built assets to an existing release:

```bash
gh release upload v1.0.0 dist/tachyon-linux-x86_64.tar.gz dist/tachyon-linux-x86_64.tar.gz.sha256
```

Avoid replacing existing public assets: publish a new patch version so pinned installs stay reproducible.
CUDA's redistributable components retain their license notices in `share/licenses/CUDA-EULA.txt` and
the header files (`CUDA-EULA.html` in official CI builds). Minimal toolkit containers may omit `EULA.txt`;
set `CUDA_EULA=/path/to/matching-license` to include an explicitly supplied notice. Source builds do not
need the bundled archive.

## Local validation

```bash
make -j8 test
python3 tests/install_test.py
tools/package.sh
python3 tests/launcher_test.py build/release/tachyon-*/bin/tachyon  # requires qemu-x86_64
TACHYON_CPU=avx512 build/release/tachyon-*/bin/tachyon selftest     # on an AVX-512 VNNI host
TACH_TEST_DEVICE=cuda python3 tests/settings_test.py build/release/tachyon-*/bin/tachyon
```

The installer tests use local fixture downloads, so they do not install into your home or need a release.
GPU checks require a compatible NVIDIA driver and GPU; the workflow's CPU runners cannot verify GPU inference.
