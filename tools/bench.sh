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
# Installed as tachyon-bench: uses the bundled binaries without building.
# TACHYON_BENCH_DIR overrides the results directory (default: user cache for installed releases).
#
# Source-tree cpu mode builds one binary per kernel set from the current tree (build/bench/), all with -mtune=native:
# avx512 (-march=native), avx2 (-march=haswell) and scalar (-march=x86-64). Sets the CPU cannot run are skipped.
# Results are printed as a markdown table; the raw runs are kept in build/bench/.
set -euo pipefail
script=$(readlink -f -- "$0")
root=$(cd -- "$(dirname "$script")/.." && pwd)
installed=0
[[ ! -x $root/libexec/tachyon-scalar ]] || installed=1
if (( ! installed )); then cd "$root"; fi

usage() { sed -n '2,/^[^#]/{/^#/p}' "$script" | sed 's/^# \{0,1\}//' >&2; exit 2; }
if [[ ${1:-} == --help || ${1:-} == -h ]]; then sed -n '2,/^[^#]/{/^#/p}' "$script" | sed 's/^# \{0,1\}//'; exit 0; fi
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
for value in "$REPS" "$T" "$P" "$N"; do
    [[ $value =~ ^[0-9]+$ && $value =~ [1-9] ]] || { echo 'Runs, threads and token counts must be positive integers' >&2; exit 2; }
done
[[ $SPEC =~ ^[0-8]$ ]] || { echo 'Speculation must be an integer from 0 to 8' >&2; exit 2; }
MODELS=("$@"); [ ${#MODELS[@]} -gt 0 ] || MODELS=("${DEFAULT_MODELS[@]}")

if (( installed )); then W=${TACHYON_BENCH_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}/tachyon/bench}
else W=${TACHYON_BENCH_DIR:-build/bench}; fi
mkdir -p "$W"
VARS=(); declare -A BIN ARGS
add_var() { VARS+=("$1"); BIN[$1]=$2; ARGS[$1]=$3; }

build() {  # build <name> <flags...>: one tachyon binary per kernel set, in its own object directory
    local name=$1; shift
    if (( installed )); then BIN[$name]="$root/libexec/tachyon-$name"; return; fi
    BIN[$name]="$W/tachyon-$name"
    echo "building $name ($*)..." >&2
    make -j"$(nproc)" BUILD="$W/obj-$name" OUT="$W/tachyon-$name" OPT="-O3 $*" tachyon >"$W/build-$name.log" 2>&1 \
        || { tail -20 "$W/build-$name.log" >&2; echo "build of $name failed (see $W/build-$name.log)" >&2; exit 1; }
}

if [ "$mode" = cpu ]; then
    cpu_flags=" $(grep -m1 '^flags' /proc/cpuinfo) "
    has() { local f; for f; do [[ $cpu_flags == *" $f "* ]] || return 1; done; }
    if has avx512f avx512bw avx512vl avx512_vnni; then build avx512 -march=native -mtune=native; add_var avx512 "${BIN[avx512]}" "--device cpu -t $T"
    else echo "skipping avx512: this CPU has no AVX-512 F/BW/VL/VNNI" >&2; fi
    if has avx2 fma f16c; then build avx2 -march=haswell -mtune=native; add_var avx2 "${BIN[avx2]}" "--device cpu -t $T"
    else echo "skipping avx2: this CPU has no AVX2/FMA/F16C" >&2; fi
    build scalar -march=x86-64 -mtune=native; add_var scalar "${BIN[scalar]}" "--device cpu -t $T"
else
    if (( installed )); then gpu_binary="$root/bin/tachyon"
    else
        gpu_binary=./tachyon
        echo "building tachyon..." >&2
        make -j"$(nproc)" tachyon >"$W/build-gpu.log" 2>&1 || { tail -20 "$W/build-gpu.log" >&2; exit 1; }
    fi
    add_var cuda "$gpu_binary" "--device cuda"
fi

# pull the models (a no-op when they are up to date) and find their files
declare -A FILE
for m in "${MODELS[@]}"; do
    "${BIN[${VARS[0]}]}" pull "$m" >&2
    FILE[$m]=$("${BIN[${VARS[0]}]}" show "$m" | sed -n 's/^[[:space:]]*file[[:space:]]*\(.*\) ([^()]*)$/\1/p')
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
awk -v variants="${VARS[*]}" -v models="${MODELS[*]}" '
function fmt(key, field,    n,i,j,v,values,median) {
    n=count[key]
    for(i=1;i<=n;i++) values[i]=sample[key,field,i]
    for(i=2;i<=n;i++) { v=values[i]; j=i-1; while(j>=1 && values[j]>v) { values[j+1]=values[j]; j-- } values[j+1]=v }
    median=n%2 ? values[(n+1)/2] : (values[n/2]+values[n/2+1])/2
    return sprintf("%.0f (%.0f-%.0f)", median, values[1], values[n])
}
{ key=$2 SUBSEP $1; n=++count[key]; sample[key,1,n]=$3; sample[key,2,n]=$4 }
END {
    nv=split(variants, vs, " "); nm=split(models, ms, " ")
    print "| Model | Config | Prefill tok/s | Decode tok/s |"
    print "|---|---|---|---|"
    for(m=1;m<=nm;m++) {
        first=1
        for(v=1;v<=nv;v++) {
            key=ms[m] SUBSEP vs[v]
            if(!count[key]) continue
            printf "| %s | %s | %s | %s |\n", first ? ms[m] : "", vs[v], fmt(key,1), fmt(key,2)
            first=0
        }
    }
}' "$raw"
echo
echo "raw runs: $raw"
