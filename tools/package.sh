#!/usr/bin/env bash
# Build a relocatable Linux x86-64 release; no GPU is needed on the build machine.
set -euo pipefail
cd "$(dirname "$0")/.."

version=${1:-$(sed -n 's/^#define TACH_VERSION "\(.*\)"/\1/p' src/common.h)}
[[ $version =~ ^[0-9]+\.[0-9]+\.[0-9]+([.-][A-Za-z0-9.-]+)?$ ]] || { echo "Invalid version: $version" >&2; exit 1; }
source_version=$(sed -n 's/^#define TACH_VERSION "\(.*\)"/\1/p' src/common.h)
[[ $version == "$source_version" ]] || { echo "Version must match TACH_VERSION ($source_version)" >&2; exit 1; }
[[ $(uname -sm) == 'Linux x86_64' ]] || { echo 'Release builds require Linux x86-64' >&2; exit 1; }

cuda=${CUDA:-/usr/local/cuda}
cxx=${CXX:-g++}
jobs=${JOBS:-$(nproc)}
work="$PWD/build/release"
name="tachyon-${version}-linux-x86_64"
stage="$work/$name"
mkdir -p "$work" "$PWD/dist"
rm -rf "$stage"
mkdir -p "$stage/bin" "$stage/libexec" "$stage/cuda/lib64" "$stage/share/licenses"

# All variants use explicit ISA flags. Static C++ runtimes avoid a dependency on the builder's libstdc++.
for cpu in scalar avx2 avx512; do
    flags='-O3 -march=x86-64 -mtune=generic -std=gnu++20 -g -Wall -Wno-unused-function -Wno-sign-compare -Wno-unused-variable -Wno-array-bounds -fno-math-errno -pthread'
    [[ $cpu == scalar ]] || flags+=' -mavx2 -mfma -mf16c'
    [[ $cpu != avx512 ]] || flags+=' -mavx512f -mavx512bw -mavx512vl -mavx512vnni'
    # Override ambient CXXFLAGS/OPT as well: a developer's -march=native must never reach a release.
    make -j"$jobs" CXX="$cxx" CUDA="$cuda" CPU="$cpu" BUILD="$work/obj-$cpu" \
        OUT="$stage/libexec/tachyon-$cpu" CLIENT_OUT="$stage/bin/tachyon-client" \
        CXXFLAGS="$flags" LDFLAGS='-static-libstdc++ -static-libgcc' tachyon
done
make -j"$jobs" CXX="$cxx" CUDA="$cuda" CPU=scalar BUILD="$work/obj-scalar" \
    CXXFLAGS='-O3 -march=x86-64 -mtune=generic -std=gnu++20 -g -pthread' \
    CLIENT_OUT="$stage/bin/tachyon-client" LDFLAGS='-static-libstdc++ -static-libgcc' "$stage/bin/tachyon-client"
"$cxx" -O2 -std=c++20 -march=x86-64 -mtune=generic -static-libstdc++ -static-libgcc \
    tools/launcher.cpp -o "$stage/bin/tachyon"
strip "$stage"/bin/* "$stage"/libexec/*

# Bundle NVRTC and its matching builtins/headers. The NVIDIA driver remains supplied by the host.
# Preserve the versioned builtins filename: NVRTC locates that exact name beside itself.
for pattern in 'libnvrtc.so*' 'libnvrtc-builtins.so*'; do
    found=0
    for library in "$cuda"/lib64/$pattern; do
        [[ -f $library ]] || continue
        real=$(readlink -f "$library")
        [[ -f $stage/cuda/lib64/$(basename "$real") ]] || cp "$real" "$stage/cuda/lib64/"
        if [[ $(basename "$library") != "$(basename "$real")" ]]; then
            ln -sf "$(basename "$real")" "$stage/cuda/lib64/$(basename "$library")"
        fi
        found=1
    done
    [[ $found == 1 ]] || { echo "Missing $pattern in $cuda/lib64" >&2; exit 1; }
done
# Keep the complete header tree so transitive CUDA includes match the bundled NVRTC version.
cp -LR "$cuda/include" "$stage/cuda/include"
cp "$cuda/EULA.txt" "$stage/share/licenses/CUDA-EULA.txt"
cp LICENSE.txt "$stage/share/licenses/Tachyon-MIT.txt"
cp README.md "$stage/README.md"
printf '%s\n' "$version" > "$stage/VERSION"
git rev-parse HEAD > "$stage/COMMIT"

# Record the actual glibc floor of this build, including bundled CUDA libraries.
floor=$(find "$stage/bin" "$stage/libexec" "$stage/cuda/lib64" -type f -exec objdump -T {} + |
    sed -n 's/.*(GLIBC_\([0-9][0-9.]*\)).*/\1/p' | sort -Vu | tail -1)
[[ -n $floor ]] || { echo 'Cannot determine glibc requirement' >&2; exit 1; }
printf '%s\n' "$floor" > "$stage/GLIBC_MIN"
"$stage/bin/tachyon" --version
TACHYON_CPU=scalar "$stage/bin/tachyon" selftest

archive="$PWD/dist/tachyon-linux-x86_64.tar.gz"
tar -C "$work" -czf "$archive" "$name"
(cd dist && sha256sum tachyon-linux-x86_64.tar.gz > tachyon-linux-x86_64.tar.gz.sha256)
echo "Created $archive (glibc >= $floor)"
