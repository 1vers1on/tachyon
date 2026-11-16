#!/bin/bash
# End-to-end benchmark: prefill and decode tok/s, as the median of several interleaved runs.
#
#   tools/bench.sh cpu [options] [model ...]   compare the CPU kernel sets (AVX-512, AVX2, scalar) the CPU can run
#   tools/bench.sh gpu [options] [model ...]   benchmark the GPU backend (CUDA)
#
# options:  -r reps       runs per model and configuration (default 5 for cpu, 3 for gpu)
#           -t threads    CPU threads (default: half of the logical cores)
#           -p tokens     prompt length (default 512 for cpu, 2048 for gpu)
#           -n tokens     generated tokens (default 128)
#           -s tokens     n-gram speculative draft tokens (default 0 = off). The benchmark prompt repeats one sentence, so
#                         with speculation on nearly every draft is accepted and decode looks 4-5x faster than it is
# models:   catalog names or Hugging Face repos, as for `tachyon pull`; each is downloaded and converted if missing.
#           Defaults: cpu: qwen3:0.6b qwen3:0.6b-q8 qwen3:1.7b   gpu: qwen3:0.6b qwen3:8b
#
# cpu mode builds one binary per kernel set from the current tree (build/bench/), all with -mtune=native:
# avx512 (-march=native), avx2 (-march=haswell) and scalar (-march=x86-64). Sets the CPU cannot run are skipped.
# Results are printed as a markdown table; the raw runs are kept in build/bench/.
set -euo pipefail
cd "$(dirname "$0")/.."

usage() { sed -n '2,/^[^#]/{/^#/p}' "$0" | sed 's/^# \{0,1\}//' >&2; exit 2; }
mode=${1:-}
case $mode in cpu|gpu) shift ;; *) usage ;; esac

if [ "$mode" = cpu ]; then REPS=5; P=512; DEFAULT_MODELS=(qwen3:0.6b qwen3:0.6b-q8 qwen3:1.7b)
else REPS=3; P=2048; DEFAULT_MODELS=(qwen3:0.6b qwen3:8b); fi
N=128; SPEC=0
T=$(( $(nproc) / 2 )); [ "$T" -ge 1 ] || T=1
while getopts r:t:p:n:s: o; do
    case $o in r) REPS=$OPTARG ;; t) T=$OPTARG ;; p) P=$OPTARG ;; n) N=$OPTARG ;; s) SPEC=$OPTARG ;; *) usage ;; esac
done
shift $((OPTIND - 1))
MODELS=("$@"); [ ${#MODELS[@]} -gt 0 ] || MODELS=("${DEFAULT_MODELS[@]}")

W=build/bench
mkdir -p "$W"
VARS=(); declare -A BIN ARGS
add_var() { VARS+=("$1"); BIN[$1]=$2; ARGS[$1]=$3; }

build() {  # build <name> <flags...>: one tachyon binary per kernel set, in its own object directory
    local name=$1; shift
    echo "building $name ($*)..." >&2
    make -j"$(nproc)" BUILD="$W/obj-$name" OUT="$W/tachyon-$name" OPT="-O3 $*" tachyon >"$W/build-$name.log" 2>&1 \
        || { tail -20 "$W/build-$name.log" >&2; echo "build of $name failed (see $W/build-$name.log)" >&2; exit 1; }
}

if [ "$mode" = cpu ]; then
    cpu_flags=" $(grep -m1 '^flags' /proc/cpuinfo) "
    has() { local f; for f; do [[ $cpu_flags == *" $f "* ]] || return 1; done; }
    if has avx512f avx512bw avx512vl avx512_vnni; then build avx512 -march=native -mtune=native; add_var avx512 "$W/tachyon-avx512" "--device cpu -t $T"
    else echo "skipping avx512: this CPU has no AVX-512 F/BW/VL/VNNI" >&2; fi
    if has avx2 fma f16c; then build avx2 -march=haswell -mtune=native; add_var avx2 "$W/tachyon-avx2" "--device cpu -t $T"
    else echo "skipping avx2: this CPU has no AVX2/FMA/F16C" >&2; fi
    build scalar -march=x86-64 -mtune=native; add_var scalar "$W/tachyon-scalar" "--device cpu -t $T"
else
    echo "building tachyon..." >&2
    make -j"$(nproc)" tachyon >"$W/build-gpu.log" 2>&1 || { tail -20 "$W/build-gpu.log" >&2; exit 1; }
    add_var cuda ./tachyon "--device cuda"
fi

# pull the models (a no-op when they are up to date) and find their files
declare -A FILE
for m in "${MODELS[@]}"; do
    "${BIN[${VARS[0]}]}" pull "$m" >&2
    FILE[$m]=$("${BIN[${VARS[0]}]}" show "$m" | awk '$1 == "file" { print $2 }')
    [ -f "${FILE[$m]}" ] || { echo "cannot find the file of $m after pulling it" >&2; exit 1; }
done

raw="$W/$mode-$(date +%Y%m%d-%H%M%S).log"
: > "$raw"
for rep in $(seq "$REPS"); do                     # interleaved, so drift in machine state hits every configuration equally
    for m in "${MODELS[@]}"; do
        for v in "${VARS[@]}"; do
            # shellcheck disable=SC2086
            r=$("${BIN[$v]}" bench -m "${FILE[$m]}" ${ARGS[$v]} -n "$N" --prompt-len "$P" --temp 0 --spec "$SPEC" 2>&1 \
                | sed -n 's/.*= \([0-9.]*\) tok\/s | gen .*= \([0-9.]*\) tok\/s.*/\1 \2/p' | tail -1) || true
            if [ -z "$r" ]; then echo "run $rep/$REPS $m $v: no result" >&2; continue; fi
            echo "$v $m $r" >> "$raw"
            echo "run $rep/$REPS $m $v: prefill/decode tok/s $r" >&2
        done
    done
done

echo
echo "CPU: $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2- | sed 's/^ //'), $(nproc) logical cores"
if [ "$mode" = gpu ] && command -v nvidia-smi >/dev/null; then echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"; fi
echo "$mode backend, prompt $P tokens, $N generated, greedy, median of $REPS runs (min-max), speculation $([ "$SPEC" = 0 ] && echo off || echo "on ($SPEC draft tokens)")$([ "$mode" = cpu ] && echo ", $T threads")"
echo
python3 -I - "$raw" "${VARS[*]}" "${MODELS[*]}" <<'EOF'
import sys, statistics as st
raw, variants, models = sys.argv[1], sys.argv[2].split(), sys.argv[3].split()
rows = {}
for line in open(raw):
    v, m, pf, dc = line.split()
    rows.setdefault((m, v), []).append((float(pf), float(dc)))
def fmt(xs):
    return f"{st.median(xs):.0f} ({min(xs):.0f}-{max(xs):.0f})"
print("| Model | Config | Prefill tok/s | Decode tok/s |")
print("|---|---|---|---|")
for m in models:
    first = True
    for v in variants:
        r = rows.get((m, v))
        if not r: continue
        print(f"| {m if first else ''} | {v} | {fmt([x[0] for x in r])} | {fmt([x[1] for x in r])} |")
        first = False
EOF
echo
echo "raw runs: $raw"
