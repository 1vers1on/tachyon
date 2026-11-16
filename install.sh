#!/usr/bin/env bash
# Usage: curl -fsSL https://raw.githubusercontent.com/1vers1on/tachyon/main/install.sh | bash
set -euo pipefail

install_tmp=''
install_staging=''
install_switch_dir=''

main() {
    local version=${TACHYON_VERSION:-latest} prefix=${TACHYON_INSTALL_DIR:-"$HOME/.local"}
    local repository=https://github.com/1vers1on/tachyon
    trap 'rm -rf -- "$install_tmp"; if [[ -n $install_staging ]]; then rm -rf -- "$install_staging"; fi; if [[ -n $install_switch_dir ]]; then rm -rf -- "$install_switch_dir"; fi' EXIT
    trap 'exit 130' INT
    trap 'exit 143' TERM
    die() { printf 'tachyon: %s\n' "$*" >&2; exit 1; }
    while (($#)); do
        case $1 in
            --version|--prefix)
                [[ $# -ge 2 && -n $2 ]] || die "$1 requires a value"
                if [[ $1 == --version ]]; then version=$2; else prefix=$2; fi
                shift 2 ;;
            -h|--help)
                cat <<'EOF'
Install Tachyon from verified GitHub release assets (Linux x86-64, glibc).

  bash install.sh [--version 1.0.1] [--prefix "$HOME/.local"]

Defaults: latest release, ~/.local (no sudo). TACHYON_VERSION and
TACHYON_INSTALL_DIR also set these options. --prefix is an installation
root containing bin/ and lib/, not the bin directory itself.
EOF
                return ;;
            *) die "unknown option: $1 (see --help)" ;;
        esac
    done
    [[ $version == latest || $version =~ ^v?[0-9]+\.[0-9]+\.[0-9]+([.-][A-Za-z0-9.-]+)?$ ]] || die 'invalid version'
    [[ -n $prefix ]] || die 'installation prefix is empty'
    [[ $(uname -s) == Linux ]] || die 'this release supports Linux (including WSL2)'
    [[ $(uname -m) == x86_64 || $(uname -m) == amd64 ]] || die 'this release requires an x86-64 CPU'
    local command
    for command in curl tar gzip mktemp getconf; do
        command -v "$command" >/dev/null || die "required command not found: $command"
    done
    local libc
    libc=$(getconf GNU_LIBC_VERSION 2>/dev/null) || die 'glibc is required; musl/Alpine binaries are not available'
    libc=${libc##* }
    [[ $libc =~ ^[0-9]+\.[0-9]+$ ]] || die 'cannot determine the installed glibc version'
    local checksum
    if command -v sha256sum >/dev/null; then checksum=sha256sum
    elif command -v shasum >/dev/null; then checksum=shasum
    else die 'sha256sum or shasum is required'; fi

    install_tmp=$(mktemp -d)
    local base="$repository/releases/latest/download"
    [[ $version == latest ]] || base="$repository/releases/download/v${version#v}"
    local asset=tachyon-linux-x86_64.tar.gz expected actual
    printf 'Downloading Tachyon (%s)...\n' "$version"
    curl -fsSL --retry 3 --connect-timeout 15 --max-time 300 --proto '=https' --proto-redir '=https' \
        -o "$install_tmp/checksum" "$base/$asset.sha256" ||
        die "release not available: $base (a maintainer must publish the binary assets first)"
    # The checksum rejects mismatched downloads if latest changes during installation.
    curl -fsSL --retry 3 --connect-timeout 15 --max-time 600 --proto '=https' --proto-redir '=https' \
        -o "$install_tmp/$asset" "$base/$asset" || die 'binary download failed'
    read -r expected _ < "$install_tmp/checksum" || die 'empty checksum file'
    [[ $expected =~ ^[0-9a-fA-F]{64}$ ]] || die 'invalid SHA-256 checksum'
    if [[ $checksum == sha256sum ]]; then actual=$(sha256sum "$install_tmp/$asset")
    else actual=$(shasum -a 256 "$install_tmp/$asset"); fi
    actual=${actual%% *}
    [[ ${actual,,} == "${expected,,}" ]] || die 'SHA-256 verification failed; installation was left untouched. Try again.'

    mkdir "$install_tmp/unpack"
    tar -xzf "$install_tmp/$asset" -C "$install_tmp/unpack" --strip-components=1 --no-same-owner --no-same-permissions || die 'cannot extract release'
    local payload="$install_tmp/unpack" required installed
    for required in bin/tachyon bin/tachyon-client libexec/tachyon-scalar libexec/tachyon-avx2 libexec/tachyon-avx512 GLIBC_MIN VERSION; do
        [[ -f $payload/$required && ! -L $payload/$required ]] || die "release is missing $required"
    done
    required=$(cat "$payload/GLIBC_MIN")
    [[ $required =~ ^[0-9]+\.[0-9]+$ ]] || die 'invalid glibc requirement in release'
    if (( ${libc%%.*} < ${required%%.*} || (${libc%%.*} == ${required%%.*} && ${libc#*.} < ${required#*.}) )); then
        die "this binary requires glibc >= $required; found $libc. Use an older build or build from source."
    fi
    installed=$(cat "$payload/VERSION")
    [[ $installed =~ ^[0-9]+\.[0-9]+\.[0-9]+([.-][A-Za-z0-9.-]+)?$ ]] || die 'invalid release version'
    [[ $version == latest || ${version#v} == "$installed" ]] || die 'release version does not match requested version'
    "$payload/bin/tachyon" --version > "$install_tmp/version" || die 'binary cannot run on this system'
    [[ $(cat "$install_tmp/version") == "$installed" ]] || die 'binary version does not match release metadata'

    mkdir -p -- "$prefix/bin" "$prefix/lib/tachyon"
    prefix=$(cd -- "$prefix" && pwd -P)
    local root="$prefix/lib/tachyon" binary
    local binaries=(tachyon tachyon-client)
    if [[ -f $payload/bin/tachyon-bench && ! -L $payload/bin/tachyon-bench ]]; then binaries+=(tachyon-bench); fi
    for binary in "${binaries[@]}"; do
        if [[ -e $prefix/bin/$binary || -L $prefix/bin/$binary ]]; then
            [[ -L $prefix/bin/$binary && $(readlink "$prefix/bin/$binary") == "$root/current/bin/$binary" ]] ||
                die "$prefix/bin/$binary already exists and is not managed by this installer; move it first"
        fi
    done
    [[ ! -e $root/current || -L $root/current ]] || die "$root/current is not an installer symlink"
    install_staging=$(mktemp -d "$root/$installed.XXXXXXXX")
    cp -a "$payload/." "$install_staging/"
    install_switch_dir=$(mktemp -d "$root/.switch.XXXXXXXX")
    ln -s "$install_staging" "$install_switch_dir/current"
    # A single rename switches the commands. Existing processes keep using their previous release.
    mv -Tf "$install_switch_dir/current" "$root/current"
    install_staging=''
    for binary in "${binaries[@]}"; do
        ln -s "$root/current/bin/$binary" "$install_switch_dir/$binary"
        mv -Tf "$install_switch_dir/$binary" "$prefix/bin/$binary"
    done
    printf 'Installed Tachyon %s to %s/bin\n' "$installed" "$prefix"
    # Print a literal $PATH for the user's shell profile.
    # shellcheck disable=SC2016
    case :$PATH: in
        *:"$prefix/bin":*) ;;
        *) printf '\nAdd this to your shell profile, then open a new shell:\n  export PATH=%q:"$PATH"\n' "$prefix/bin" ;;
    esac
    printf '\nGet started: tachyon run qwen3:0.6b\n'
}

# Keep invocation last: a truncated piped download cannot execute a partial installer.
main "$@"
