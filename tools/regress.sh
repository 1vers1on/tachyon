#!/bin/bash
# Regression perplexities (wikitext-2 test, 2 windows): both CUDA paths for the standard model set.
# usage: tools/regress.sh [binary] [extra models...]     e.g. tools/regress.sh ./tachyon models/olmoe-1b-7b-q4.tach
BIN=${1:-./tachyon}; [ $# -gt 0 ] && shift
MODELS="models/qwen3-0.6b-q4.tach models/lfm2.5-230m-q4.tach models/gemma-3-270m-it-q8.tach models/qwen3.5-0.8b-q8.tach $*"
N=${N:-2}
for m in $MODELS; do
  [ -f "$m" ] || { printf "%-28s skipped (not found)\n" "$(basename $m .tach)"; continue; }
  a=$($BIN ppl -q -m $m -f data/wiki.test.raw -n $N -c 1024 --slots 1 2>&1 | grep -o 'perplexity: [0-9.]*' | cut -d' ' -f2)
  b=$(TACH_PPL_CHUNK=8 $BIN ppl -q -m $m -f data/wiki.test.raw -n $N -c 1024 --slots 1 2>&1 | grep -o 'perplexity: [0-9.]*' | cut -d' ' -f2)
  printf "%-28s prefill %-10s decode %-10s\n" "$(basename $m .tach)" "${a:-FAIL}" "${b:-FAIL}"
done
